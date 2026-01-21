# Real-Time Fraud Detection Engine

## Executive Summary
This project is intended to implement a real-time fraud detection system for high-velocity fintech environments (e.g. Wealthsimple, KOHO). Potential economic impact will be used as the guiding purpose of this project. An economic cost function will be used to balance the cost of fraud (chargebacks, loss of funds) with the cost of customer friction (churn risk, support costs).

## Business Value & ROI
Financial institutions lose approx. 3-5% of revenue to fraud (CITE SOURCE). Blocking legitimate transactions (False Positives) insults customers and drives churn, which can be more expensive in the long run. This project has three goals to bring to the organization:
* **High Latency -** <50ms P95 inference time to ensure zero friction at checkout.
* **Dynamism -** Dynamic thresholds based on `E[Cost] = FN_cost + FP_cost`.
* **Reliable Architecture -** Hybrid Lambda Architecture using Redis (Real-time) and PostgreSQL/dbt (Batch).