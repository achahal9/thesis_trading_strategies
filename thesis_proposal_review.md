# Thesis Proposal Review: Trading on Information
## *Fusing Alternative Data Signals and Supply-Chain Propagation in the German Equity Market*

**Authors:** Jean-Christophe Scheider & Arjun Chahal  
**Reviewer Role:** Financial Literature Reviewer & Mentor  
**Date:** March 25, 2026

---

## Overview Assessment

This is an **ambitious and well-structured** proposal that combines alternative data signal construction with graph neural network (GNN)-based supply chain modeling in the German equity market. The core idea—that information shocks propagate through revenue-concentrated supply chains with a delay that can be captured by relational ML models—is timely, theoretically grounded, and commercially relevant.

However, the proposal's scope is its biggest risk. Below, I raise critical questions across six dimensions and then provide answers and recommendations for each.

---

## 1. Conceptual & Theoretical Foundations

### Questions Raised

| # | Question |
|---|---------|
| Q1.1 | **EMH Framing:** You invoke Fama (1970) and then argue anomalies exist—but which *form* of EMH are you challenging? Semi-strong? If so, insider filings (WpHG) are public information post-filing, so your argument needs to be about *speed of incorporation*, not *availability*. Are you arguing for slow information diffusion or limits to arbitrage? |
| Q1.2 | **Conrad & Kaul (1998) connection:** You cite their decomposition framework to justify "rigorous signal attribution," but their paper is really about whether momentum profits come from cross-sectional variation in expected returns vs. time-series predictability. How exactly does their framework inform your signal construction or testing methodology? |
| Q1.3 | **Missing theoretical mechanism for GNN propagation:** Why *specifically* would information from a supplier's insider purchases propagate to a customer's stock price? Is this based on real-options theory (the supplier's insider knows about a big order)? Or attention/limited processing? The theoretical channel is underspecified. |
| Q1.4 | **Fama-French (2015) is cited but not used:** You reference the five-factor model in your references but use the Carhart four-factor model for risk adjustment. Why not use the five-factor model, which adds profitability (RMW) and investment (CMA) factors—potentially relevant for industrial supply chain firms? |

### Answers & Recommendations

> **Q1.1 Answer:** You are effectively challenging the **semi-strong form** of market efficiency, but the mechanism is **limits to arbitrage** (Shleifer & Vishny, 1997) combined with **investor inattention** (Hirshleifer & Teoh, 2003; Cohen & Frazzini, 2008). The argument should be: even though WpHG filings are public, (a) MDAX/SDAX stocks have low analyst coverage, so fewer sophisticated investors monitor these disclosures, and (b) the *second-order implications* (impact on supply chain neighbors) require processing capacity that most market participants lack. I recommend explicitly framing this as an "investor inattention" story, citing Cohen & Frazzini (2008) on customer-supplier return predictability.

> **Q1.2 Answer:** The Conrad & Kaul citation is somewhat tangential. Their key insight is that holding-period matters for strategy evaluation. This is relevant for your *backtest design* (monthly rebalancing vs. quarterly), not really for signal construction. **Recommendation:** Either make this connection explicit—"Following Conrad & Kaul (1998), we evaluate strategy profitability across multiple holding horizons to disentangle cross-sectional predictability from time-series momentum"—or replace it with a more directly relevant citation like Jegadeesh & Titman (1993) for momentum, or Cohen & Frazzini (2008) for supply-chain return predictability.

> **Q1.3 Answer:** The theoretical channel should be explicitly stated. The most compelling framing is: **information asymmetry + network complexity = delayed price discovery.** When an insider at Firm A purchases shares, it signals private positive information about Firm A's prospects. If Firm B derives 30% of its revenue from Firm A, then Firm A's good news is mechanically good news for Firm B—but the market doesn't connect these dots instantly because (a) the supply chain link isn't salient to most investors, and (b) Firm B's analyst coverage may not overlap with Firm A's. **Recommendation:** Add 2–3 sentences explicitly laying out this causal chain. Also cite Menzly & Ozbas (2010, "Market Segmentation and Cross-Predictability of Returns") as direct theoretical support.

> **Q1.4 Answer:** Using Carhart (1997) four-factor is defensible for a German market study because constructing reliable RMW and CMA factors for Germany is data-intensive and the factors may not be as well-established in smaller-cap German equities. **Recommendation:** Acknowledge this explicitly. State that you use Carhart as your primary model but will include a robustness check using a five-factor specification if data permits.

---

## 2. Hypotheses: Clarity, Testability & Overlap

### Questions Raised

| # | Question |
|---|---------|
| Q2.1 | **H1 and H3 overlap significantly.** H1 says "Analyst earnings revision momentum *and* WpHG insider purchases are positively associated with subsequent abnormal returns." H3 says "WpHG insider purchase filings are positively associated with subsequent abnormal returns." H3 is a strict subset of H1. Why are they separate hypotheses? |
| Q2.2 | **H1 bundles two distinct signals.** Analyst revision momentum and insider purchases are fundamentally different information sources (sell-side vs. corporate insiders). Shouldn't these be separate hypotheses? |
| Q2.3 | **H4 is vague.** "A hybrid GNN-XGBoost strategy will outperform individual signals and traditional benchmarks." Outperform by what metric? Over what time period? What is the null model? This is not a testable hypothesis as written. |
| Q2.4 | **H5 is the most novel hypothesis but lacks a clear statistical test.** How will you define "extreme positive alternative signals"? Top decile? 2 standard deviations? What is "delayed but statistically significant"? What is the event window? |
| Q2.5 | **No null/negative hypothesis.** Where is the hypothesis that short interest (H2) combined with network effects might amplify downside? This seems like a natural extension of H5 for the short leg. |

### Answers & Recommendations

> **Q2.1–Q2.2 Answer:** The hypotheses should be restructured as follows:
> - **H1a:** Analyst revision momentum is positively associated with subsequent abnormal returns, with the effect amplified in MDAX/SDAX segments.
> - **H1b:** WpHG insider net purchases are positively associated with subsequent abnormal returns, with the effect amplified in MDAX/SDAX segments.
> - **H2:** Elevated short interest ratios are negatively associated with subsequent returns.
> - **H3 (Network Alpha):** Abnormal signal strength at a firm's major customers/suppliers predicts delayed abnormal returns at the focal firm.
> - **H4 (Model Superiority):** A hybrid GNN-XGBoost model generates statistically significant Carhart alpha net of transaction costs, outperforming the best single-signal long-short portfolio.
> 
> This eliminates redundancy and makes each hypothesis independently testable.

> **Q2.3 Answer:** Restate H4 as: "The composite ML signal portfolio generates a statistically significant (t > 2.0) monthly Carhart four-factor alpha over the out-of-sample period 2019–2026, and this alpha exceeds the best-performing univariate signal portfolio by at least [X] basis points per month."

> **Q2.4 Answer:** Define operationally: "We define 'extreme positive signals' as observations in the top decile of the cross-sectional signal distribution. The 'delayed' effect is measured over a [1, 3]-month window following the signal observation at the network neighbor." This gives you a testable event-study-like framework.

> **Q2.5 Answer:** Consider adding: **H3b (Negative Network Alpha):** Extreme negative signals at network neighbors (high short interest, earnings misses) predict delayed negative returns at the focal firm. This makes the long-short portfolio construction symmetric and theoretically consistent.

---

## 3. Data & Signal Construction

### Questions Raised

| # | Question |
|---|---------|
| Q3.1 | **Survivorship bias:** You use DAX/MDAX/SDAX constituents from 2006–2026. These indices rebalance regularly. Are you using *current* constituents (survivorship bias) or *historical* constituents (point-in-time)? This is a critical design choice. |
| Q3.2 | **WpHG data availability:** You specify WpHG Section 15a—but this was reformed into MAR (Market Abuse Regulation) Article 19 in July 2016 when the EU regulation superseded national law. Are you accounting for the structural break in reporting requirements and thresholds? |
| Q3.3 | **Short interest data for Germany:** Bloomberg's short interest data for German equities is notoriously patchy before ~2012, and the EU Short Selling Regulation (236/2012) only mandated public disclosure of net short positions above 0.5% of issued share capital from November 2012. How will you handle pre-2012 data gaps? |
| Q3.4 | **Graph construction:** "Edges are weighted by revenue concentration." From where? Bloomberg's SPLC <GO> function provides some supplier/customer data, but it is incomplete, US-centric, and updated irregularly. How will you construct a *time-varying* German supply chain graph with sufficient coverage for MDAX/SDAX firms? |
| Q3.5 | **Signal lagging:** You lag all signals by one full month. For insider filings (which must be disclosed within 3 business days under MAR), a one-month lag is conservative but may sacrifice signal strength. Have you considered a shorter lag (e.g., T+5 trading days) with a sensitivity analysis? |

### Answers & Recommendations

> **Q3.1 Answer:** This is the **single most important methodological decision** in the paper. You must use **point-in-time index constituents.** Bloomberg provides historical index membership via the MEMB <GO> function. If you use current constituents, your results will be contaminated by survivorship bias—firms that were delisted, went bankrupt, or were demoted from MDAX to SDAX will be excluded, systematically biasing returns upward. **Recommendation:** Explicitly state "We use point-in-time index membership, reconstructed from Bloomberg's historical constituent data, to avoid survivorship and look-ahead bias."

> **Q3.2 Answer:** This is a real structural break. Pre-July 2016, German insider reporting was governed by WpHG §15a with national-specific thresholds; post-July 2016, it falls under EU MAR Article 19 with harmonized rules (e.g., the €5,000 annual threshold before disclosure is required was raised to €20,000). **Recommendation:** Acknowledge this regime change. Include a dummy variable for the MAR transition in your Fama-MacBeth regressions, or split your analysis into pre-MAR (2006–2016) and post-MAR (2017–2026) subperiods.

> **Q3.3 Answer:** Pre-2012, you have a data coverage problem. **Recommendation:** Either (a) start your analysis in 2012 when short interest data becomes reliable (losing the GFC period but gaining data integrity), or (b) keep 2006–2026 but clearly flag that short interest is only available for a subset of the period, and test H2 separately for the 2012–2026 window.

> **Q3.4 Answer:** This is arguably the **hardest data challenge** in the entire thesis. Bloomberg SPLC provides supply chain mappings but with known limitations: US-centric coverage, lagged updates, and incomplete MDAX/SDAX coverage. **Recommendation:** Be transparent about this limitation. Consider supplementing Bloomberg SPLC with annual report segment disclosures (firms are required under IFRS 8 to report revenue by segment/geography if >10% of total revenue). Also consider a simpler approach: use industry-level input-output tables from Destatis (German Federal Statistical Office) as a proxy for inter-firm linkages. This is less granular but more complete.

> **Q3.5 Answer:** The one-month lag is conservative and appropriate for a monthly rebalancing strategy. However, **Recommendation:** Include a robustness check with a T+5 trading day lag for the insider signal specifically, since the informational content of insider purchases decays quickly (see Seyhun, 1986; Lakonishok & Lee, 2001).

---

## 4. Methodology: ML Architecture & Statistical Testing

### Questions Raised

| # | Question |
|---|---------|
| Q4.1 | **GNN architecture underspecified.** What type of GNN? GCN? GAT? GraphSAGE? The choice matters significantly for how neighborhood information is aggregated. What are the hyperparameters? How many layers (too many = over-smoothing)? |
| Q4.2 | **"Hybrid GNN-XGBoost" is unclear.** How do these two models interact? Is the GNN a feature extractor (producing node embeddings that are fed into XGBoost)? Or is it an ensemble (averaging GNN and XGBoost predictions)? These are fundamentally different architectures. |
| Q4.3 | **Rolling expanding window:** You specify training 2006–2018, testing 2019–2026. But is this a *single* train-test split or a *rolling* expanding window that retrains periodically? If it's a single split, you have a problem: the model never sees COVID or 2022 energy crisis data during training. |
| Q4.4 | **Overfitting risk:** With 4 signals × ~300 firms × 20 years = potentially thin cross-sections, how will you prevent the XGBoost from overfitting, especially with GNN-generated features? What regularization? What cross-validation scheme? |
| Q4.5 | **Fama-MacBeth + ML:** There's a tension between your univariate testing (Fama-MacBeth, a linear framework) and your ML modeling (non-linear). How do you reconcile a finding that a signal is insignificant in Fama-MacBeth but highly important in XGBoost's SHAP values? |
| Q4.6 | **Transaction costs:** You mention "net of estimated transaction costs" but don't specify how you'll estimate them. For MDAX/SDAX stocks, bid-ask spreads and market impact can be substantial. |

### Answers & Recommendations

> **Q4.1 Answer:** For a supply chain graph where edges have meaningful weights (revenue concentration), a **Graph Attention Network (GAT)** or **GraphSAGE** is preferable to a vanilla GCN. GAT can learn attention weights over neighbors, which aligns with your hypothesis that some supply chain links matter more than others. **Recommendation:** Specify that you will use a 2-layer GAT with edge-weight initialization. Cite Veličković et al. (2018) for GAT. More than 2 layers risks over-smoothing (all nodes converge to similar embeddings), which would destroy firm-level signal variation.

> **Q4.2 Answer:** Clarify the architecture. The most defensible approach for a thesis is: **GNN as feature extractor → XGBoost as predictor.** The GNN produces node embeddings that capture each firm's supply-chain context. These embeddings are concatenated with the original 4 signals and fed into XGBoost. This is interpretable (SHAP can decompose both raw signals and network features) and avoids end-to-end training complexity. **Recommendation:** Add one paragraph clearly specifying this pipeline.

> **Q4.3 Answer:** Use a **rolling expanding window with periodic retraining.** For example: initial training on 2006–2018, predict 2019. Then retrain on 2006–2019, predict 2020. And so forth. This is standard in empirical asset pricing ML (Gu, Kelly, Xiu, 2020 use exactly this approach). **Recommendation:** State this explicitly and cite Gu et al. (2020) for the methodology.

> **Q4.4 Answer:** Regularization is critical. **Recommendation:** (a) XGBoost: use early stopping with a validation set carved from the training window, limit tree depth to 4–6, and set a minimum child weight. (b) GNN: use dropout on the attention coefficients and L2 weight decay. (c) Report results with and without GNN features to quantify the marginal contribution of the network layer, which also serves as an overfitting check.

> **Q4.5 Answer:** This is actually a *feature*, not a bug. Fama-MacBeth tests linear, unconditional predictability. XGBoost captures conditional, non-linear effects. A signal that is insignificant in Fama-MacBeth but important in XGBoost suggests it has predictive power only in interaction with other signals or in specific market regimes. **Recommendation:** Frame this explicitly as a contribution: "Our multi-stage approach first establishes linear baseline predictability via Fama-MacBeth, then uses ML to capture incremental non-linear alpha."

> **Q4.6 Answer:** For German equities, use a tiered transaction cost model: DAX stocks: 10 bps round-trip; MDAX: 20–30 bps; SDAX: 40–60 bps. These can be estimated from Bloomberg's quoted bid-ask spreads (BEST_BID1/BEST_ASK1 historical data). **Recommendation:** Add a sentence specifying your transaction cost assumptions and cite Novy-Marx & Velikov (2016) or Frazzini et al. (2018) for methodology.

---

## 5. Scope, Feasibility & Timeline

### Questions Raised

| # | Question |
|---|---------|
| Q5.1 | **Scope creep risk:** This proposal contains (a) four separate signal constructions, (b) univariate Fama-MacBeth regressions for each, (c) supply chain graph construction, (d) a GNN implementation, (e) an XGBoost model, (f) SHAP analysis, (g) a portfolio backtest, and (h) robustness checks. This is at least 2–3 separate papers' worth of work. Is this achievable in ~4 months (April–July 2026)? |
| Q5.2 | **Phase 1 bottleneck:** Bloomberg data extraction for 20 years of data across 3 indices with 4 signal types is a major undertaking. Analyst revision history, short interest, earnings announcements, and insider filings each require different Bloomberg functions and may have different data coverage. Is one month realistic? |
| Q5.3 | **GNN training complexity:** Phase 3 (May–June) includes GNN training, XGBoost training, and SHAP attribution *and* a thesis draft submission by June 1st. This seems extremely compressed. Have you prototyped the GNN pipeline? |
| Q5.4 | **Two authors:** The proposal lists two authors. How will the work be divided? Is there a risk that the GNN component becomes one person's work while the other handles signals, creating integration challenges? |

### Answers & Recommendations

> **Q5.1 Answer:** This is a legitimate concern. **Recommendation:** Define a **minimum viable thesis (MVT):** Signals + Fama-MacBeth + XGBoost (no GNN) + simple long-short backtest. The GNN/network alpha component is the "stretch goal." This way, even if the GNN doesn't work or time runs out, you have a complete, publishable thesis on alternative signal predictability in the German market.

> **Q5.2 Answer:** Bloomberg data extraction is always slower than expected. Excel API pulls time out, BQL has query limits, and point-in-time data requires careful dating. **Recommendation:** Start Phase 1 *immediately* (before April if possible). Prioritize getting price data and analyst revision data first (these are the easiest), then insider filings, then short interest. Build data validation scripts from day one.

> **Q5.3 Answer:** The June 1st draft deadline with simultaneous GNN development is very tight. **Recommendation:** Have the XGBoost-only model running by May 15th and write the thesis around that. If the GNN works by June 1st, include it. If not, the thesis stands on signals + XGBoost + Fama-MacBeth, and GNN becomes a "future work" section.

> **Q5.4 Answer:** **Recommendation:** Clearly define workstreams: Person A handles signal construction + Fama-MacBeth; Person B handles ML pipeline + graph construction. Both collaborate on the backtest and writing. Hold weekly integration checkpoints.

---

## 6. Literature & References

### Questions Raised

| # | Question |
|---|---------|
| Q6.1 | **Critical missing citations:** Cohen & Frazzini (2008), "Economic Links and Predictable Returns"—this is THE foundational paper on customer-supplier return predictability and is not cited. |
| Q6.2 | **Missing:** Jegadeesh & Titman (1993) on momentum—you discuss momentum signals but don't cite the foundational paper. |
| Q6.3 | **Missing:** Lakonishok & Lee (2001) or Seyhun (1986) on insider trading predictability. These directly support your WpHG signal. |
| Q6.4 | **Missing:** Shleifer & Vishny (1997) on limits to arbitrage—needed to explain why mispricings persist in the German mid/small-cap space. |
| Q6.5 | **Bloomberg Intelligence (2025) is not peer-reviewed.** It's a sell-side report. You can cite it for motivation, but don't rely on it as your primary academic justification for the network hypothesis. |
| Q6.6 | **Background word count:** The guidelines specify max 400 words for the background. Your background section appears to exceed this significantly. Similarly, methodology is limited to 250 words but yours is substantially longer. |

### Answers & Recommendations

> **Q6.1–Q6.4 Answer:** These are essential additions. Your reference list should include:
> - Cohen, L. & Frazzini, A. (2008). Economic Links and Predictable Returns. *Journal of Finance*, 63(4), 1977–2011.
> - Jegadeesh, N. & Titman, S. (1993). Returns to Buying Winners and Selling Losers. *Journal of Finance*, 48(1), 65–91.
> - Lakonishok, J. & Lee, I. (2001). Are Insider Trades Informative? *Review of Financial Studies*, 14(1), 79–111.
> - Shleifer, A. & Vishny, R. (1997). The Limits of Arbitrage. *Journal of Finance*, 52(1), 35–55.
> - Hirshleifer, D. & Teoh, S.H. (2003). Limited Attention, Information Disclosure, and Financial Reporting. *Journal of Accounting and Economics*, 36(1–3), 337–386.
> - Menzly, L. & Ozbas, O. (2010). Market Segmentation and Cross-Predictability of Returns. *Journal of Finance*, 65(4), 1555–1580.

> **Q6.5 Answer:** Use Bloomberg Intelligence as *supporting evidence* but ground the network hypothesis in Cohen & Frazzini (2008) and Menzly & Ozbas (2010). These are published in the *Journal of Finance* and provide the academic foundation.

> **Q6.6 Answer:** Check your word counts carefully. The proposal guidelines (from your README) are clear: Background max 400 words, Methodology max 250 words. If you're over, you'll need to tighten. Move detailed methodology into the appendix or save it for the thesis itself. The proposal should be a concise signal of intent, not a miniature thesis.

---

## Summary: Strengths & Priority Actions

### ✅ Strengths
1. **Original contribution:** GNN-based supply chain signal propagation in the German market is genuinely novel
2. **Strong institutional knowledge:** WpHG/MAR regulatory awareness shows domain expertise
3. **Interpretability focus:** SHAP decomposition elevates this above "black box ML" critiques
4. **Multi-regime sample:** 2006–2026 covers GFC, Euro crisis, COVID, and energy crisis
5. **Research question is well-scoped** for the German market gap

### 🔴 Priority Actions (Ranked)

| Priority | Action | Impact |
|----------|--------|--------|
| **P0** | Fix survivorship bias methodology (point-in-time constituents) | Fatal flaw if not addressed |
| **P0** | Address WpHG→MAR regulatory break in data | Undermines insider signal validity |
| **P1** | Restructure hypotheses to eliminate overlap (H1/H3) | Evaluator will flag redundancy |
| **P1** | Add missing foundational citations (Cohen & Frazzini, Jegadeesh & Titman) | Credibility gap without them |
| **P1** | Check word count compliance (400/250 limits) | May violate submission rules |
| **P2** | Specify GNN architecture and hybrid pipeline | Methodological clarity |
| **P2** | Define minimum viable thesis (MVT) fallback plan | Risk management |
| **P2** | Address short interest data gaps pre-2012 | Data integrity |
| **P3** | Add transaction cost model specification | Backtest credibility |
| **P3** | Clarify Conrad & Kaul citation relevance | Minor but noticeable |

---

> [!IMPORTANT]
> The proposal has strong bones—the research question is compelling, the German market gap is real, and the GNN angle is genuinely innovative. The main risks are **scope** (too much for one thesis) and **data** (supply chain graph construction for German mid-caps is very hard). Define your MVT early, start Bloomberg data extraction immediately, and you'll be in good shape.
