import { useEffect, useState } from "react";
import "./App.css";

const editableFields = [
  {
    key: "AMT_INCOME_TOTAL",
    label: "Annual Income",
    type: "number",
    prefix: "$",
    min: 0,
    step: 1000,
  },
  {
    key: "AMT_CREDIT",
    label: "Credit Amount",
    type: "number",
    prefix: "$",
    min: 0,
    step: 1000,
  },
  {
    key: "AMT_ANNUITY",
    label: "Annuity",
    type: "number",
    prefix: "$",
    min: 0,
    step: 100,
  },
  {
    key: "AMT_GOODS_PRICE",
    label: "Goods Price",
    type: "number",
    prefix: "$",
    min: 0,
    step: 1000,
  },
  {
    key: "AGE_YEARS",
    label: "Age",
    type: "number",
    min: 18,
    max: 100,
    step: 1,
  },
  {
    key: "EMPLOYED_YEARS",
    label: "Employment Years",
    type: "number",
    min: 0,
    max: 80,
    step: 1,
  },
  {
    key: "EXT_SOURCE_1",
    label: "External Score 1",
    type: "number",
    min: 0,
    max: 1,
    step: 0.01,
  },
  {
    key: "EXT_SOURCE_2",
    label: "External Score 2",
    type: "number",
    min: 0,
    max: 1,
    step: 0.01,
  },
  {
    key: "EXT_SOURCE_3",
    label: "External Score 3",
    type: "number",
    min: 0,
    max: 1,
    step: 0.01,
  },
];

const categoricalFields = [
  {
    key: "CODE_GENDER",
    label: "Gender",
    options: [
      { value: "M", label: "Male" },
      { value: "F", label: "Female" },
      { value: "XNA", label: "Unknown / XNA" },
    ],
  },
  {
    key: "FLAG_OWN_CAR",
    label: "Owns Car",
    options: [
      { value: "Y", label: "Yes" },
      { value: "N", label: "No" },
    ],
  },
  {
    key: "FLAG_OWN_REALTY",
    label: "Owns Property",
    options: [
      { value: "Y", label: "Yes" },
      { value: "N", label: "No" },
    ],
  },
  {
    key: "NAME_EDUCATION_TYPE",
    label: "Education",
    options: [
      {
        value: "Secondary / secondary special",
        label: "Secondary / Secondary Special",
      },
      {
        value: "Higher education",
        label: "Higher Education",
      },
      {
        value: "Incomplete higher",
        label: "Incomplete Higher",
      },
      {
        value: "Lower secondary",
        label: "Lower Secondary",
      },
      {
        value: "Academic degree",
        label: "Academic Degree",
      },
    ],
  },
  {
    key: "NAME_INCOME_TYPE",
    label: "Income Type",
    options: [
      { value: "Working", label: "Working" },
      { value: "State servant", label: "State Servant" },
      { value: "Commercial associate", label: "Commercial Associate" },
      { value: "Pensioner", label: "Pensioner" },
      { value: "Unemployed", label: "Unemployed" },
      { value: "Student", label: "Student" },
      { value: "Businessman", label: "Businessman" },
      { value: "Maternity leave", label: "Maternity Leave" },
    ],
  },
  {
    key: "NAME_HOUSING_TYPE",
    label: "Housing Type",
    options: [
      { value: "House / apartment", label: "House / Apartment" },
      { value: "Rented apartment", label: "Rented Apartment" },
      { value: "With parents", label: "With Parents" },
      { value: "Municipal apartment", label: "Municipal Apartment" },
      { value: "Office apartment", label: "Office Apartment" },
      { value: "Co-op apartment", label: "Co-op Apartment" },
    ],
  },
];

function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [baselineApplicant, setBaselineApplicant] = useState(null);
  const [applicant, setApplicant] = useState(null);
  const [lgd, setLgd] = useState(0.40);
  const [baselineResult, setBaselineResult] = useState(null);

  useEffect(() => {
  const loadApplicant = async () => {
    try {
      const response = await fetch("/sample_applicant.json");

      if (!response.ok) {
        throw new Error("Could not load the example applicant.");
      }

      const data = await response.json();
      const applicantData = Array.isArray(data) ? data[0] : data;

      setBaselineApplicant(applicantData);
      setApplicant({ ...applicantData });
    } catch (err) {
      setError(err.message);
    }
  };

  loadApplicant();
}, []);

const requestAssessment = async (applicantData, lgdValue) => {
  const response = await fetch("http://127.0.0.1:5000/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      applicant: applicantData,
      lgd: lgdValue,
    }),
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.error || "Prediction failed.");
  }

  return data;
};

const scoreApplicant = async () => {
  try {
    setLoading(true);
    setError(null);

    if (!applicant) {
      throw new Error("Applicant data has not loaded yet.");
    }

    // Validate editable applicant fields
    for (const field of editableFields) {
      const value = applicant[field.key];

      if (value === null || value === undefined || value === "") {
        throw new Error(`${field.label} must have a value.`);
      }

      if (field.min !== undefined && value < field.min) {
        throw new Error(
          `${field.label} must be at least ${field.min}.`
        );
      }

      if (field.max !== undefined && value > field.max) {
        throw new Error(
          `${field.label} must not exceed ${field.max}.`
        );
      }
    }

    // Score the current scenario.
    const data = await requestAssessment(applicant, lgd);

    // Store a fixed baseline using the original applicant and 40% LGD.
    if (!baselineResult && baselineApplicant) {
      const baselineData = await requestAssessment(baselineApplicant, 0.40);
      setBaselineResult(baselineData);
    }

    setResult(data);
  } catch (err) {
    setError(err.message);
  } finally {
    setLoading(false);
  }
};

  const formatPercent = (value) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  const formatMoney = (value) => {
    return new Intl.NumberFormat("en-AU", {
      style: "currency",
      currency: "AUD",
      maximumFractionDigits: 2,
    }).format(value);
  };

  const handleApplicantChange = (key, value) => {
  setApplicant((previous) => ({
    ...previous,
    [key]: value === "" ? null : Number(value),
  }));
  };

  const handleCategoricalChange = (key, value) => {
  setApplicant((previous) => ({
    ...previous,
    [key]: value,
  }));
};

  const resetApplicant = () => {
    setApplicant({ ...baselineApplicant });
    setLgd(0.40);
    setResult(null);
    setError(null);
  };

  return (
    <div className="app">
      <header className="header">
        <div>
          <div className="title-row">
            <h1>Credit Risk Assessment</h1>
            <span className="prototype-badge">Research Prototype</span>
          </div>
          <p>Hybrid Ensemble Credit Scoring & Scenario Analysis Framework</p>
        </div>

        <div className="status">
          <span className="status-dot"></span>
          Model Ready
        </div>
      </header>

      <main className="dashboard">
        <section className="config-panel">
  <div className="config-header">
    <div>
      <span className="section-label">APPLICATION</span>
      <h2>Applicant Scenario</h2>

      <p>
        Modify selected characteristics to explore alternative credit-risk scenarios.
        Variables not shown retain the original baseline applicant values.
      </p>
    </div>
  </div>

  <div className="input-grid">
    {editableFields.map((field) => (
      <div className="input-group" key={field.key}>
        <label htmlFor={field.key}>{field.label}</label>

        <div className="input-wrapper">
          {field.prefix && (
            <span className="input-prefix">{field.prefix}</span>
          )}

              <input
                id={field.key}
                type={field.type}
                min={field.min}
                max={field.max}
                step={field.step || 1}
                value={applicant?.[field.key] ?? ""}
                onChange={(event) =>
                  handleApplicantChange(field.key, event.target.value)
                }
              />
        </div>

        <small>{field.key}</small>
      </div>
    ))}
  </div>

<div className="input-section-header">
  <h3>Applicant Information</h3>
  <p>
    Select categorical characteristics to evaluate how the model responds
    under alternative applicant scenarios.
  </p>
</div>

<div className="input-grid">
  {categoricalFields.map((field) => (
    <div className="input-group" key={field.key}>
      <label htmlFor={field.key}>{field.label}</label>

      <select
        id={field.key}
        value={applicant?.[field.key] ?? ""}
        onChange={(event) =>
          handleCategoricalChange(field.key, event.target.value)
        }
      >
        {field.options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>

      <small>{field.key}</small>
    </div>
  ))}
</div>

<div className="input-section-header">
  <h3>Financial Assumptions</h3>
  <p>
    Adjust the Loss Given Default assumption used for financial analysis.
    LGD affects Expected Loss and the indicative rate, but not the model PD.
  </p>
</div>

<div className="financial-assumption-grid">
  <div className="input-group">
    <label htmlFor="lgd">Loss Given Default (LGD)</label>
    <select
      id="lgd"
      value={lgd}
      onChange={(event) => setLgd(Number(event.target.value))}
    >
      <option value={0.20}>20% — Lower-loss scenario</option>
      <option value={0.40}>40% — Base scenario</option>
      <option value={0.60}>60% — Higher-loss scenario</option>
    </select>
    <small>Financial assumption — not a model feature</small>
  </div>
</div>

  <div className="scenario-actions">
    <button
      className="predict-button"
      onClick={scoreApplicant}
      disabled={!applicant || loading}
    >
      {loading ? "Assessing..." : "Assess Scenario"}
    </button>

    <button
      className="reset-button"
      onClick={resetApplicant}
      disabled={!baselineApplicant || loading}
    >
      Reset to Baseline
    </button>
  </div>

  {error && <div className="error-message">{error}</div>}
</section>

        {!result && (
          <section className="empty-state">
            <div className="empty-icon">◎</div>
            <h2>No assessment generated</h2>
            <p>
              Configure the applicant above and select Assess Scenario
              to generate probability of default and financial risk estimates.
            </p>
          </section>
        )}

        {result && (
          <>
            <section>
              <div className="section-title">
                <span className="section-label">RISK OUTPUT</span>
                <h2>Credit Risk Assessment</h2>
              </div>

              <div className="metric-grid">
                <div className="metric-card primary-metric">
                  <span>Calibrated PD</span>
                  <strong>{formatPercent(result.final_pd)}</strong>
                  <small>Probability of Default</small>
                </div>

                <div className="metric-card">
                  <span>Expected Loss</span>
                  <strong>{formatMoney(result.expected_loss)}</strong>
                  <small>Scenario-based estimate</small>
                </div>

                <div className="metric-card">
                  <span>Indicative Rate</span>
                  <strong>{formatPercent(result.indicative_rate)}</strong>
                  <small>Illustrative risk-adjusted rate</small>
                </div>

                <div className="metric-card">
                  <span>Exposure</span>
                  <strong>{formatMoney(result.ead_proxy)}</strong>
                  <small>AMT_CREDIT used as EAD proxy</small>
                </div>
              </div>
            </section>

            {baselineResult && (
              <section className="card">
                <div className="section-title">
                  <span className="section-label">SCENARIO COMPARISON</span>
                  <h2>Baseline vs Current Scenario</h2>
                </div>

                <p className="comparison-description">
                  Baseline results use the original example applicant and 40% LGD.
                  Changes in PD and indicative rate are reported in percentage points.
                </p>

                <div className="comparison-table">
                  <div className="comparison-row comparison-header">
                    <span>Output</span>
                    <span>Baseline</span>
                    <span>Scenario</span>
                    <span>Change</span>
                  </div>

                  <ComparisonRow label="Calibrated PD" baseline={baselineResult.final_pd} scenario={result.final_pd} type="percent" />
                  <ComparisonRow label="Expected Loss" baseline={baselineResult.expected_loss} scenario={result.expected_loss} type="money" />
                  <ComparisonRow label="Indicative Rate" baseline={baselineResult.indicative_rate} scenario={result.indicative_rate} type="percent" />
                </div>
              </section>
            )}

            <section className="card">
              <div className="section-title">
                <span className="section-label">ENSEMBLE</span>
                <h2>Model Predictions</h2>
              </div>

              <div className="model-list">
                <ModelRow
                  name="Logistic Regression"
                  value={result.lr_pd}
                />

                <ModelRow
                  name="XGBoost"
                  value={result.xgb_pd}
                />

                <ModelRow
                  name="Neural Network (MLP)"
                  value={result.mlp_pd}
                />

                <ModelRow
                  name="Raw Ensemble"
                  value={result.raw_ensemble_pd}
                />

                <ModelRow
                  name="Final Calibrated PD"
                  value={result.final_pd}
                  finalModel
                />
              </div>
            </section>
            
            <section className="card">
                <div className="section-title">
                    <span className="section-label">EXPLAINABILITY</span>
                    <h2>Applicant Risk Drivers</h2>
                </div>

                <p className="explanation-description">
                    Top local SHAP contributions from the XGBoost component for this
                    applicant. Positive values push the XGBoost model output higher,
                    while negative values push it lower.
                </p>

                <div className="shap-list">
                    {result.xgb_explanation?.map((item, index) => (
                    <div className="shap-row" key={`${item.feature}-${index}`}>
                        <div className="shap-rank">
                        {index + 1}
                        </div>

                        <div className="shap-feature">
                        <strong>{item.feature}</strong>
                        <span>{item.direction}</span>
                        </div>

                        <div
                        className={
                            item.shap_value >= 0
                            ? "shap-value positive"
                            : "shap-value negative"
                        }
                        >
                        {item.shap_value >= 0 ? "+" : ""}
                        {item.shap_value.toFixed(4)}
                        </div>
                    </div>
                    ))}
                </div>

                <div className="shap-note">
                    <strong>Interpretation note:</strong> These SHAP values explain the
                    XGBoost component of the ensemble. They do not directly decompose
                    the final calibrated probability of default and should not be
                    interpreted as percentage-point changes in PD.
                </div>
                </section>

            <section className="card financial-card">
              <div className="section-title">
                <span className="section-label">FINANCIAL ANALYSIS</span>
                <h2>Expected Loss Scenario</h2>
              </div>

              <div className="financial-equation">
                <div>
                  <span>PD</span>
                  <strong>{formatPercent(result.final_pd)}</strong>
                </div>

                <span className="operator">×</span>

                <div>
                  <span>LGD</span>
                  <strong>{formatPercent(result.lgd)}</strong>
                </div>

                <span className="operator">×</span>

                <div>
                  <span>EAD Proxy</span>
                  <strong>{formatMoney(result.ead_proxy)}</strong>
                </div>

                <span className="operator">=</span>

                <div className="equation-result">
                  <span>Expected Loss</span>
                  <strong>{formatMoney(result.expected_loss)}</strong>
                </div>
              </div>

              <p className="disclaimer">
                Research prototype only. Financial outputs are scenario-based estimates.
                AMT_CREDIT is used as an EAD proxy. The indicative rate is illustrative
                and is not an actual lending offer or credit decision.
              </p>
            </section>

            

          </>
        )}
        <footer className="app-footer">
          Academic research prototype · Hybrid ensemble credit scoring framework
        </footer>
      </main>
    </div>
  );
}

function ComparisonRow({ label, baseline, scenario, type }) {
  const change = scenario - baseline;

  const formatValue = (value) => {
    if (type === "money") {
      return new Intl.NumberFormat("en-AU", {
        style: "currency",
        currency: "AUD",
        maximumFractionDigits: 2,
      }).format(value);
    }
    return `${(value * 100).toFixed(2)}%`;
  };

  const formatChange = () => {
    if (type === "money") {
      const sign = change > 0 ? "+" : "";
      return `${sign}${new Intl.NumberFormat("en-AU", {
        style: "currency",
        currency: "AUD",
        maximumFractionDigits: 2,
      }).format(change)}`;
    }

    const percentagePoints = change * 100;
    const sign = percentagePoints > 0 ? "+" : "";
    return `${sign}${percentagePoints.toFixed(2)} pp`;
  };

  return (
    <div className="comparison-row">
      <strong>{label}</strong>
      <span>{formatValue(baseline)}</span>
      <span>{formatValue(scenario)}</span>
      <span className={change > 0 ? "comparison-up" : change < 0 ? "comparison-down" : ""}>
        {formatChange()}
      </span>
    </div>
  );
}

function ModelRow({ name, value, finalModel = false }) {
  const percentage = value * 100;

  return (
    <div className={`model-row ${finalModel ? "final-model" : ""}`}>
      <div className="model-info">
        <span>{name}</span>
        <strong>{percentage.toFixed(2)}%</strong>
      </div>

      <div className="progress-track">
        <div
          className="progress-fill"
          style={{ width: `${Math.min(percentage, 100)}%` }}
        ></div>
      </div>
    </div>
  );
}

export default App;