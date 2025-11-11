# score_service/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
import os
import openai
import uvicorn
from datetime import datetime

openai.api_key = os.getenv("OPENAI_API_KEY")  # GPT-4o-mini key

app = FastAPI(title="CrediLens Scoring Service")

class Transaction(BaseModel):
    date: str
    description: str
    amount: float
    type: str
    balance: float | None = None

class ScoreRequest(BaseModel):
    transactions: List[Transaction]
    fraud_score: float = 0.0
    fraud_flags: List[str] = []
    credit_score_external: float | None = None

# ---------------- Metric functions ----------------
def group_by_month(transactions):
    # returns dict month -> list of tx amounts, credits, debits, closing balance
    by_month = {}
    for t in transactions:
        # expect date as YYYY-MM-DD or MM/DD/YYYY
        try:
            dt = datetime.fromisoformat(t.date)
        except Exception:
            try:
                dt = datetime.strptime(t.date, "%m/%d/%Y")
            except:
                dt = None
        key = dt.strftime("%Y-%m") if dt else "unknown"
        by_month.setdefault(key, []).append(t)
    return by_month

def calc_income_metrics(transactions):
    months = group_by_month(transactions)
    monthly_income = []
    for m, txs in months.items():
        income = sum(t.amount for t in txs if t.type=="credit")
        monthly_income.append(income)
    avg_income = np.mean(monthly_income) if monthly_income else 0.0
    vol = float(np.std(monthly_income)/max(np.mean(monthly_income),1)) if len(monthly_income)>1 else 0.0
    diversity = len(set([t.description for t in transactions if t.type=="credit"])) 
    trend = 0.0
    if len(monthly_income) >= 2 and monthly_income[0] != 0:
        trend = (monthly_income[-1] - monthly_income[0]) / max(monthly_income[0],1)
    return {"avg_income": round(float(avg_income),2), "income_volatility": round(vol,3), "income_diversity": diversity, "income_trend": round(trend,3)}

def calc_cashflow_metrics(transactions):
    months = group_by_month(transactions)
    monthly_savings_rate = []
    avg_balances = []
    negative_balance_incidents = 0
    for m, txs in months.items():
        credits = sum(t.amount for t in txs if t.type=="credit")
        debits = sum(t.amount for t in txs if t.type=="debit")
        if credits > 0:
            monthly_savings_rate.append(max(0.0, (credits - debits)/credits))
        # approximate closing balance
        balances = [t.balance for t in txs if t.balance is not None]
        if balances:
            avg_balances.append(balances[-1])
            negative_balance_incidents += sum(1 for b in balances if b < 0)
    savings_rate = float(np.mean(monthly_savings_rate)) if monthly_savings_rate else 0.0
    liquidity = (np.mean(avg_balances)/ (np.mean([sum(t.amount for t in txs if t.type=="debit") for txs in months.values()]) + 1)) if avg_balances else 0.0
    neg_freq = negative_balance_incidents / max(len(transactions),1)
    return {"savings_rate": round(savings_rate,3), "liquidity_buffer": round(liquidity,2), "negative_balance_freq": round(neg_freq,3)}

def calc_spending_metrics(transactions):
    debits = [t.amount for t in transactions if t.type=="debit"]
    total_spend = sum(debits) if debits else 1.0
    essentials_keywords = ["rent","bond","eskom","municipal","school","tuition","insurance"]
    essential_spend = sum(t.amount for t in transactions if t.type=="debit" and any(k in t.description.lower() for k in essentials_keywords))
    subs_keywords = ["netflix","spotify","dstv","showmax","subscription","monthly"]
    subs_spend = sum(t.amount for t in transactions if t.type=="debit" and any(k in t.description.lower() for k in subs_keywords))
    essential_ratio = essential_spend/total_spend if total_spend else 0.0
    subscription_burden = subs_spend/max(np.mean([sum(t.amount for t in transactions if t.type=="credit")]) ,1) if debits else 0.0
    discretionary_control = 1 - (np.std(debits)/max(np.mean(debits),1)) if debits else 0.5
    return {"essential_spending_ratio": round(essential_ratio,3), "subscription_burden": round(subscription_burden,3), "discretionary_control": round(discretionary_control,3)}

def calc_responsibility_metrics(transactions):
    balances = [t.balance for t in transactions if t.balance is not None]
    overdraft_days = sum(1 for b in balances if b<0)
    overdraft_avoidance = 1 - (overdraft_days / max(len(balances),1)) if balances else 0.8
    positive_saving_months = sum(1 for m,txs in group_by_month(transactions).items() if sum(t.amount for t in txs if t.type=="credit") > sum(t.amount for t in txs if t.type=="debit"))
    saving_consistency = positive_saving_months / max(len(group_by_month(transactions)),1)
    bill_timely = 0.9  # placeholder: needs bill due date info; keep conservative
    return {"overdraft_avoidance": round(overdraft_avoidance,3), "saving_consistency": round(saving_consistency,3), "bill_punctuality": round(bill_timely,3)}

# ---------------- GPT reason code generation ----------------
def generate_reason_codes(metrics: Dict[str,Any], pd: float, credit_score_external: float | None, fraud_flags: List[str]):
    # craft concise prompt; keep token usage small with structured JSON output
    prompt_lines = [
        "You are an explainability assistant for a credit scoring system.",
        "Input: JSON with behavioral submetrics, PD (probability of default between 0 and 1), optional external credit score, and fraud flags.",
        "Output: JSON with keys: primary_reasons (list), improvement_suggestions (list), confidence (0-1).",
        "Do not include raw metric values in the primary reasons; express them in plain language. Keep suggestions short."
    ]
    input_payload = {
        "metrics": metrics,
        "PD": pd,
        "external_score": credit_score_external,
        "fraud_flags": fraud_flags
    }
    prompt = "\n".join(prompt_lines) + "\n\nINPUT:\n" + str(input_payload) + "\n\nReturn JSON only."

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=300,
            temperature=0.0
        )
        text = resp["choices"][0]["message"]["content"]
        import json, re
        m = re.search(r'\{.*\}', text, flags=re.S)
        if m:
            return json.loads(m.group(0))
        else:
            return {"primary_reasons": ["Insufficient data for explanation"], "improvement_suggestions": [], "confidence": 0.6}
    except Exception as e:
        return {"primary_reasons": ["Reason generation failed"], "improvement_suggestions": [], "confidence": 0.5}

# ---------------- Main scoring endpoint ----------------
@app.post("/score")
async def score_endpoint(req: ScoreRequest):
    txs = req.transactions
    if not txs:
        raise HTTPException(status_code=400, detail="No transactions provided")

    # compute groups of metrics
    income = calc_income_metrics(txs)
    cash = calc_cashflow_metrics(txs)
    spend = calc_spending_metrics(txs)
    resp = calc_responsibility_metrics(txs)

    # normalize and combine to composite scores
    income_stab = max(0, 1 - income["income_volatility"])
    cash_health = min(1, cash["savings_rate"] + min(1, cash["liquidity_buffer"]/3))
    spending_beh = (1 - spend["subscription_burden"]) * spend["discretionary_control"]
    responsibility = (resp["overdraft_avoidance"] + resp["saving_consistency"] + resp["bill_punctuality"]) / 3

    # Weighted behavioral index
    behavioral_index = (
        income_stab * 0.28 +
        cash_health * 0.28 +
        spending_beh * 0.22 +
        responsibility * 0.22
    )

    # incorporate fraud score as penalty
    fraud_adj = max(0, 1 - (req.fraud_score / 100) * 0.4)  # fraud reduces effective behavior score
    effective_behavior = behavioral_index * fraud_adj

    # incorporate external credit if present
    bureau_weight = 0.35 if req.credit_score_external else 0.0
    bureau_component = (req.credit_score_external / 1000) if req.credit_score_external else 0.0

    combined = effective_behavior * (1 - bureau_weight) + bureau_component * bureau_weight

    # PD mapping: logistic-style but simpler for stability
    pd = float(np.clip(1 - combined, 0.01, 0.95))

    # credit score on 1000 scale
    credit_score = int(1000 - (pd * 900))

    # generate human-readable reason codes via GPT-4o-mini
    reason_payload = generate_reason_codes(
        metrics = {
            "income": income, "cash": cash, "spend": spend, "responsibility": resp,
            "behavioral_index": round(float(behavioral_index),3),
            "effective_behavior": round(float(effective_behavior),3)
        },
        pd = pd,
        credit_score_external = req.credit_score_external,
        fraud_flags = req.fraud_flags
    )

    return {
        "PD": round(pd,3),
        "credit_score": credit_score,
        "behavioral_index": round(float(behavioral_index),3),
        "fraud_adj": round(float(fraud_adj),3),
        "combined_score": round(float(combined),3),
        "metrics": {
            "income": income, "cash": cash, "spend": spend, "responsibility": resp
        },
        "reason_codes": reason_payload
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
