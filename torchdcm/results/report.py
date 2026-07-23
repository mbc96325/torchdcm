from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from html import escape
from importlib import metadata
import json
from math import erfc, isfinite, sqrt
from pathlib import Path
import platform
from statistics import NormalDist
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


def _package_version() -> str:
    try:
        return metadata.version("torchdcm")
    except metadata.PackageNotFoundError:
        return "0.1.1"


def _as_python(value: Any) -> Any:
    # Normalize tensors and NumPy scalars at the report boundary so JSON, HTML,
    # text, and LaTeX renderers consume the same presentation-neutral data.
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _as_python(value.tolist())
    if isinstance(value, (Path, torch.device, torch.dtype)):
        return str(value).removeprefix("torch.")
    if isinstance(value, dict):
        return {str(key): _as_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_python(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        return None
    return value


def _dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    clean = frame.astype(object).where(pd.notna(frame), None)
    return [_as_python(record) for record in clean.to_dict(orient="records")]


def _format_number(value: Any) -> str:
    if value is None:
        return "--"
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if bool(value) else "No"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if not isfinite(value):
            return "--"
        magnitude = abs(value)
        if magnitude != 0 and (magnitude < 1e-4 or magnitude >= 1e6):
            return f"{value:.4e}"
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        if not value:
            return "None"
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        if not value:
            return "None"
        return "; ".join(f"{key}: {item}" for key, item in value.items())
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value).removeprefix("torch.")
    return str(value)


def _parameter_group(name: str, model: object) -> str:
    upper = name.upper()
    random_coefficients = list(getattr(model, "random_coefficients", []))
    random_names = {coefficient.name for coefficient in random_coefficients}
    sigma_names = {coefficient.sigma_name or f"SIGMA_{coefficient.name}" for coefficient in random_coefficients}
    if upper.startswith("CHOL_"):
        return "Cholesky parameters"
    if name in sigma_names or upper.startswith("SIGMA_"):
        return "Random-coefficient scales"
    if name in random_names:
        return "Random-coefficient means"
    if upper.startswith("LAMBDA_"):
        return "Nest parameters"
    if upper.startswith(("TAU_", "THRESHOLD_", "CUT_")):
        return "Threshold parameters"
    if upper.startswith(("CLASS_", "MEMBERSHIP_")):
        return "Class-membership parameters"
    return "Utility coefficients"


def _null_value(name: str) -> float:
    return 1.0 if name.upper().startswith("LAMBDA_") else 0.0


@dataclass
class EstimationReport:
    """Presentation-neutral report for one fitted choice model."""

    title: str
    sections: OrderedDict[str, OrderedDict[str, Any]]
    parameters: pd.DataFrame
    covariance: pd.DataFrame
    correlation: pd.DataFrame
    alternatives: pd.DataFrame
    warnings: list[str]
    cov_type: str
    confidence_level: float

    @classmethod
    def from_results(
        cls,
        results: object,
        *,
        cov_type: str | None = None,
        confidence_level: float = 0.95,
        title: str | None = None,
    ) -> "EstimationReport":
        if not 0.0 < confidence_level < 1.0:
            raise ValueError("confidence_level must lie strictly between zero and one.")
        selected_cov = cov_type or results.cov_type
        covariance_tensor = results.cov_params(selected_cov)
        covariance_array = covariance_tensor.detach().cpu().numpy()
        parameter_names = list(results.param_names)
        covariance = pd.DataFrame(covariance_array, index=parameter_names, columns=parameter_names)
        scale = np.sqrt(np.clip(np.diag(covariance_array), a_min=0.0, a_max=None))
        denominator = np.outer(scale, scale)
        # Zero-variance parameters have undefined correlations; retain NaN so
        # renderers can display "--" instead of an artificial zero.
        correlation_array = np.divide(
            covariance_array,
            denominator,
            out=np.full_like(covariance_array, np.nan, dtype=float),
            where=denominator > 0,
        )
        correlation = pd.DataFrame(correlation_array, index=parameter_names, columns=parameter_names)

        warnings: list[str] = []
        sections: OrderedDict[str, OrderedDict[str, Any]] = OrderedDict()
        model_name = results.model.__class__.__name__
        report_title = title or f"TorchDCM {model_name} Estimation Report"
        sections["Model and run information"] = _model_run_section(results)
        sections["Data summary"] = _data_section(results.data)
        specification = _model_specification(results.model)
        if specification:
            sections["Model specification"] = specification
        sections["Estimation and convergence"] = _estimation_section(results, warnings)
        sections["Model fit"] = _fit_section(results)
        sections["Inference"] = _inference_section(results, selected_cov, confidence_level)

        # Build each table once, then reuse it verbatim across all output
        # formats to prevent HTML/text/JSON reports from drifting.
        parameters = _parameter_table(results, selected_cov, confidence_level)
        alternatives = _alternative_table(results.data)
        _append_numerical_warnings(results, parameters, covariance_array, warnings)
        _append_model_warnings(results, parameters, warnings)
        return cls(
            title=report_title,
            sections=sections,
            parameters=parameters,
            covariance=covariance,
            correlation=correlation,
            alternatives=alternatives,
            warnings=list(dict.fromkeys(warnings)),
            cov_type=selected_cov,
            confidence_level=confidence_level,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "title": self.title,
            "covariance_type": self.cov_type,
            "confidence_level": self.confidence_level,
            "sections": _as_python(self.sections),
            "parameters": _dataframe_records(self.parameters),
            "alternatives": _dataframe_records(self.alternatives),
            "covariance": {
                "parameter_names": list(self.covariance.columns),
                "values": _as_python(self.covariance.to_numpy()),
            },
            "correlation": {
                "parameter_names": list(self.correlation.columns),
                "values": _as_python(self.correlation.to_numpy()),
            },
            "warnings": list(self.warnings),
        }

    def to_text(self) -> str:
        lines = [self.title, "=" * min(96, len(self.title))]
        for section_name, values in self.sections.items():
            lines.extend(["", section_name, "-" * len(section_name)])
            label_width = min(34, max((len(label) for label in values), default=0))
            for label, value in values.items():
                lines.append(f"{label:<{label_width}}  {_format_number(value)}")
        if not self.alternatives.empty:
            lines.extend(["", "Alternative summary", "-------------------"])
            lines.append(self.alternatives.to_string(index=False, formatters=_text_formatters(self.alternatives), na_rep="--"))
        lines.extend(["", "Parameter estimates", "-------------------"])
        lines.extend(_parameter_text(self.parameters, self.confidence_level))
        if not self.covariance.empty:
            lines.extend(["", f"Variance-covariance matrix ({self.cov_type})", "-" * (29 + len(self.cov_type))])
            lines.append(_matrix_text(self.covariance))
        if not self.correlation.empty:
            lines.extend(["", f"Parameter correlation matrix ({self.cov_type})", "-" * (30 + len(self.cov_type))])
            lines.append(_matrix_text(self.correlation))
        if self.warnings:
            lines.extend(["", "Warnings", "--------"])
            lines.extend(f"- {warning}" for warning in self.warnings)
        return "\n".join(lines)

    def to_html(self) -> str:
        section_html = []
        for name, values in self.sections.items():
            rows = "".join(
                f"<tr><th>{escape(label)}</th><td>{escape(_format_number(value))}</td></tr>"
                for label, value in values.items()
            )
            wide = " wide" if name in {"Model specification", "Estimation and convergence", "Model fit"} else ""
            section_html.append(
                f"<section class='card{wide}'><h2>{escape(name)}</h2>"
                f"<table class='summary'>{rows}</table></section>"
            )
        alternatives_html = ""
        if not self.alternatives.empty:
            alternatives_html = (
                "<section class='report-table'><h2>Alternative summary</h2>"
                + self.alternatives.to_html(index=False, border=0, na_rep="--", float_format=lambda value: f"{value:.6g}")
                + "</section>"
            )
        parameter_html = self.parameters.to_html(
            index=False,
            border=0,
            na_rep="--",
            float_format=lambda value: f"{value:.6g}",
        )
        warnings_html = ""
        if self.warnings:
            warnings_html = "<section class='warnings'><h2>Warnings</h2><ul>" + "".join(
                f"<li>{escape(warning)}</li>" for warning in self.warnings
            ) + "</ul></section>"
        covariance_html = ""
        if not self.covariance.empty:
            covariance_html = (
                f"<section class='report-table'><h2>Variance-covariance matrix ({escape(self.cov_type)})</h2>"
                "<div class='matrix'>"
                + self.covariance.to_html(border=0, na_rep="--", float_format=lambda value: f"{value:.6g}")
                + "</div></section>"
            )
        correlation_html = ""
        if not self.correlation.empty:
            correlation_html = (
                f"<section class='report-table'><h2>Parameter correlation matrix ({escape(self.cov_type)})</h2>"
                "<div class='matrix'>"
                + self.correlation.to_html(border=0, na_rep="--", float_format=lambda value: f"{value:.6g}")
                + "</div></section>"
            )
        return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(self.title)}</title>
<style>
* {{ box-sizing: border-box; }}
:root {{ color-scheme: light; }}
body {{ background: #f2f5f9; color: #17202a; font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; }}
.page {{ margin: 0 auto; max-width: 1180px; padding: 34px; }}
.report-header {{ background: linear-gradient(125deg, #173b64, #2f6da3); border-radius: 14px; color: white; margin-bottom: 24px; padding: 28px 32px; }}
.kicker {{ font-size: 12px; font-weight: 700; letter-spacing: .12em; opacity: .82; text-transform: uppercase; }}
h1 {{ font-size: 30px; letter-spacing: -.02em; margin: 7px 0 5px; }}
.subtitle {{ margin: 0; opacity: .84; }}
h2 {{ color: #244f79; font-size: 17px; margin: 0 0 12px; }}
.section-grid {{ display: grid; gap: 16px; grid-template-columns: repeat(2, minmax(0, 1fr)); }}
.card, .report-table {{ background: white; border: 1px solid #dce4ed; border-radius: 10px; break-inside: avoid; margin: 0; padding: 18px 20px; }}
.card.wide {{ grid-column: 1 / -1; }}
.report-table {{ margin-top: 16px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid #e3e9ef; padding: 7px 8px; text-align: right; vertical-align: top; }}
tr:last-child th, tr:last-child td {{ border-bottom: 0; }}
thead th {{ background: #eaf1f7; color: #173b64; font-size: 12px; font-weight: 700; }}
td:first-child, th:first-child {{ text-align: left; }}
table.summary th {{ color: #435568; font-weight: 600; text-align: left; width: 46%; }}
table.summary td {{ text-align: left; }}
.report-table .dataframe {{ font-size: 12px; white-space: nowrap; }}
.matrix {{ overflow-x: auto; }}
.warnings {{ background: #fff7e8; border: 1px solid #f2d49c; border-left: 5px solid #d9822b; border-radius: 8px; break-inside: avoid; margin-top: 16px; padding: 14px 18px; }}
.footer {{ color: #687887; font-size: 12px; margin: 24px 2px 0; }}
@media (max-width: 760px) {{ .section-grid {{ grid-template-columns: 1fr; }} .card.wide {{ grid-column: auto; }} .page {{ padding: 16px; }} }}
@media print {{
  @page {{ size: Letter; margin: 0.45in; }}
  body {{ background: white; font-size: 10px; }}
  .page {{ max-width: none; padding: 0; }}
  .report-header {{ border-radius: 8px; margin-bottom: 12px; padding: 16px 20px; }}
  h1 {{ font-size: 22px; }}
  h2 {{ font-size: 13px; margin-bottom: 6px; }}
  .section-grid {{ gap: 8px; }}
  .card, .report-table {{ border-radius: 6px; padding: 9px 11px; }}
  .report-table {{ margin-top: 8px; }}
  th, td {{ padding: 3px 4px; }}
  .report-table .dataframe {{ font-size: 8px; }}
}}
</style>
</head>
<body>
<div class="page">
<header class="report-header">
  <div class="kicker">TorchDCM · Estimation output</div>
  <h1>{escape(self.title)}</h1>
  <p class="subtitle">Complete single-model results, diagnostics, and inference summary</p>
</header>
<main>
<div class="section-grid">{''.join(section_html)}</div>
{alternatives_html}
<section class="report-table"><h2>Parameter estimates</h2>{parameter_html}</section>
{covariance_html}
{correlation_html}
{warnings_html}
</main>
<p class="footer">Generated from a structured TorchDCM estimation result.</p>
</div>
</body>
</html>
"""

    def to_latex(self) -> str:
        parts = [f"\\subsection*{{{_latex_escape(self.title)}}}"]
        for name, values in self.sections.items():
            frame = pd.DataFrame(
                {"Statistic": list(values.keys()), "Value": [_format_number(value) for value in values.values()]}
            )
            parts.append(f"\\paragraph{{{_latex_escape(name)}.}}")
            parts.append(frame.to_latex(index=False, escape=True))
        if not self.alternatives.empty:
            parts.append("\\paragraph{Alternative summary.}")
            parts.append(self.alternatives.to_latex(index=False, escape=True, float_format=lambda value: f"{value:.6g}"))
        parts.append("\\paragraph{Parameter estimates.}")
        parts.append(self.parameters.to_latex(index=False, escape=True, float_format=lambda value: f"{value:.6g}"))
        if not self.covariance.empty:
            parts.append(f"\\paragraph{{Variance-covariance matrix ({_latex_escape(self.cov_type)}).}}")
            parts.append(self.covariance.to_latex(escape=True, float_format=lambda value: f"{value:.6g}"))
        if not self.correlation.empty:
            parts.append(f"\\paragraph{{Parameter correlation matrix ({_latex_escape(self.cov_type)}).}}")
            parts.append(self.correlation.to_latex(escape=True, float_format=lambda value: f"{value:.6g}"))
        if self.warnings:
            parts.append("\\paragraph{Warnings.}")
            parts.append("\\begin{itemize}")
            parts.extend(f"\\item {_latex_escape(warning)}" for warning in self.warnings)
            parts.append("\\end{itemize}")
        return "\n".join(parts)

    def save(
        self,
        directory: str | Path,
        *,
        formats: Iterable[str] = ("html", "json", "csv", "latex", "text"),
    ) -> dict[str, list[Path]]:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        aliases = {"txt": "text", "tex": "latex"}
        requested = {aliases.get(item.lower(), item.lower()) for item in formats}
        supported = {"html", "json", "csv", "latex", "text"}
        unknown = requested - supported
        if unknown:
            raise ValueError(f"Unsupported report formats: {sorted(unknown)}")

        written: dict[str, list[Path]] = {}
        if "html" in requested:
            path = target / "report.html"
            path.write_text(self.to_html(), encoding="utf-8")
            written["html"] = [path]
        if "json" in requested:
            path = target / "result.json"
            path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            written["json"] = [path]
        if "text" in requested:
            path = target / "summary.txt"
            path.write_text(self.to_text() + "\n", encoding="utf-8")
            written["text"] = [path]
        if "latex" in requested:
            path = target / "report.tex"
            path.write_text(self.to_latex() + "\n", encoding="utf-8")
            written["latex"] = [path]
        if "csv" in requested:
            csv_paths = []
            tables = {
                "parameters.csv": self.parameters,
                "alternatives.csv": self.alternatives,
                "covariance.csv": self.covariance,
                "correlation.csv": self.correlation,
            }
            for filename, frame in tables.items():
                path = target / filename
                frame.to_csv(path, index=filename in {"covariance.csv", "correlation.csv"})
                csv_paths.append(path)
            written["csv"] = csv_paths
        return written


def _model_run_section(results: object) -> OrderedDict[str, Any]:
    model = results.model
    status = results.convergence_status
    model_name = model.__class__.__name__
    objective = "Maximum simulated log likelihood" if "Mixed" in model_name else "Maximum log likelihood"
    section = OrderedDict(
        [
            ("Model family", model_name),
            ("Estimation objective", objective),
            ("TorchDCM version", _package_version()),
            ("PyTorch version", torch.__version__),
            ("Python version", platform.python_version()),
            ("Operating system", platform.platform()),
            ("Device", getattr(model, "device", getattr(results.params, "device", "cpu"))),
            ("Tensor dtype", getattr(model, "dtype", getattr(results.params, "dtype", None))),
            ("Optimizer", status.get("optimizer", "Not recorded")),
            ("Maximum iterations", getattr(model, "max_iter", None)),
            ("Line search", getattr(model, "line_search_fn", None)),
            ("Random seed", getattr(model, "seed", None)),
        ]
    )
    return OrderedDict((key, value) for key, value in section.items() if value is not None)


def _data_section(data: object) -> OrderedDict[str, Any]:
    structure = "Not applicable"
    obs_ptr = getattr(data, "obs_ptr", None)
    if obs_ptr is not None and len(obs_ptr) > 1:
        widths = torch.diff(obs_ptr.detach().cpu())
        # Equal row-pointer differences identify the vectorizable balanced case.
        structure = "Balanced" if bool((widths == widths[0]).all()) else "Ragged"
    weighted = None
    weights = getattr(data, "weights", None)
    if weights is not None:
        cpu_weights = weights.detach().cpu()
        weighted = not bool(torch.allclose(cpu_weights, torch.ones_like(cpu_weights)))
    widths = torch.diff(obs_ptr.detach().cpu()) if obs_ptr is not None and len(obs_ptr) > 1 else None
    availability = getattr(data, "availability", None)
    available_rows = int(availability.detach().cpu().sum()) if availability is not None else None
    unavailable_rows = int((~availability.detach().cpu()).sum()) if availability is not None else None
    section = OrderedDict(
        [
            ("Individuals", getattr(data, "n_individuals", None)),
            ("Choice observations", getattr(data, "n_obs", None)),
            ("Alternative rows", getattr(data, "n_rows", None)),
            ("Available alternative rows", available_rows),
            ("Unavailable alternative rows", unavailable_rows),
            ("Alternatives", getattr(data, "n_alternatives", None)),
            ("Minimum choice-set size", int(widths.min()) if widths is not None else None),
            ("Mean choice-set size", float(widths.double().mean()) if widths is not None else None),
            ("Maximum choice-set size", int(widths.max()) if widths is not None else None),
            ("Choice-set structure", structure),
            ("Panel data", bool(getattr(data, "has_panel", False))),
            ("Weighted estimation", weighted),
            ("Sum of observation weights", float(cpu_weights.sum()) if weights is not None else None),
            ("Alternative-level variables", list(getattr(data, "x_alt", {}).keys())),
            ("Observation-level variables", list(getattr(data, "x_obs", {}).keys())),
        ]
    )
    return OrderedDict((key, value) for key, value in section.items() if value is not None)


def _model_specification(model: object) -> OrderedDict[str, Any]:
    section: OrderedDict[str, Any] = OrderedDict()
    spec = getattr(model, "spec", None)
    if spec is not None:
        utilities = getattr(spec, "utilities", {})
        section["Utility functions"] = {
            alternative: _format_utility_expression(expression)
            for alternative, expression in utilities.items()
        }
        section["Utility alternatives"] = list(utilities.keys())
        parameters = list(getattr(spec, "parameters", []))
        section["Utility parameters"] = len(parameters)
        section["Estimated utility parameters"] = sum(not bool(parameter.fixed) for parameter in parameters)
        section["Fixed utility parameters"] = sum(bool(parameter.fixed) for parameter in parameters)
        section["Utility starting values"] = {parameter.name: parameter.init for parameter in parameters}
    nests = getattr(model, "nests", None)
    if nests:
        nest_members = {}
        cross_nest_allocations = {}
        for name, nest in nests.items():
            alternatives = getattr(nest, "alternatives", None)
            allocations = getattr(nest, "allocations", None)
            if alternatives is not None:
                nest_members[name] = list(alternatives)
            elif allocations is not None:
                nest_members[name] = list(allocations)
                cross_nest_allocations[name] = dict(allocations)
            else:
                nest_members[name] = []
        section["Nests"] = nest_members
        if cross_nest_allocations:
            section["Cross-nest allocations"] = cross_nest_allocations
    random_coefficients = getattr(model, "random_coefficients", None)
    if random_coefficients:
        section["Random coefficients"] = {
            coefficient.name: coefficient.distribution for coefficient in random_coefficients
        }
        section["Correlated coefficients"] = bool(getattr(model, "correlated", False))
        section["Simulation draws"] = getattr(model, "n_draws", None)
        section["Antithetic draws"] = getattr(model, "antithetic", None)
        section["Panel integration"] = getattr(model, "panel", None)
    return section


def _estimation_section(results: object, warnings: list[str]) -> OrderedDict[str, Any]:
    status = results.convergence_status
    gradient_norm = status.get("gradient_norm")
    normalized_gradient = status.get("normalized_gradient_norm")
    normalized_tolerance = status.get("normalized_gradient_tolerance")
    tolerance = status.get(
        "gradient_tolerance",
        getattr(results.model, "tolerance_grad", None),
    )
    explicit_success = status.get("success")
    success = explicit_success
    message = status.get("message")
    if success is None and gradient_norm is not None and tolerance is not None:
        message = message or "Completed (optimizer status unavailable)"
    optimizer_warnings = list(status.get("warnings", []))
    warnings.extend(optimizer_warnings)
    if success is False and not optimizer_warnings:
        warnings.append(message or "The optimizer did not report successful convergence.")
    if success is True:
        status_label = "Converged"
    elif success is False:
        status_label = "Check required"
    elif gradient_norm is not None and isfinite(float(gradient_norm)):
        status_label = "Completed"
    else:
        status_label = "Not recorded"

    information = results.hessian.detach().cpu().numpy()
    # Floating-point Hessian evaluation can introduce tiny asymmetry; symmetrize
    # before eigenvalue, rank, and condition-number diagnostics.
    information = 0.5 * (information + information.T)
    rank = int(np.linalg.matrix_rank(information)) if np.isfinite(information).all() else None
    minimum = maximum = condition = positive_definite = None
    if information.size and np.isfinite(information).all():
        eigenvalues = np.linalg.eigvalsh(information)
        minimum = float(eigenvalues.min())
        maximum = float(eigenvalues.max())
        positive_definite = bool(minimum > 0)
        absolute = np.abs(eigenvalues)
        condition = float(absolute.max() / absolute.min()) if absolute.min() > 0 else float("inf")
    section = OrderedDict(
        [
            ("Status", status_label),
            ("Termination reason", message),
            ("Optimizer iterations", status.get("optimizer_iterations")),
            ("Closure evaluations", status.get("closure_evaluations")),
            ("Function evaluations", status.get("function_evaluations")),
            ("Internal gradient infinity norm", gradient_norm),
            ("Normalized gradient infinity norm", normalized_gradient),
            ("Normalized gradient warning threshold", normalized_tolerance),
            ("Gradient tolerance", tolerance),
            ("Function/step tolerance", status.get("function_step_tolerance")),
            ("Information-matrix rank", rank),
            ("Information-matrix dimension", information.shape[0] if information.ndim == 2 else None),
            ("Positive definite", positive_definite),
            ("Smallest eigenvalue", minimum),
            ("Largest eigenvalue", maximum),
            ("Condition number", condition),
            ("Simulation draws", status.get("n_draws")),
            ("Panel likelihood", status.get("panel")),
            ("Clusters", status.get("n_clusters")),
            ("Compilation time (s)", status.get("compile_seconds")),
            ("Optimization time (s)", status.get("optimization_seconds")),
            ("Inference time (s)", status.get("inference_seconds")),
            ("Total time (s)", status.get("total_seconds")),
        ]
    )
    return OrderedDict((key, value) for key, value in section.items() if value is not None)


def _fit_section(results: object) -> OrderedDict[str, Any]:
    likelihood_ratio = max(0.0, 2.0 * (results.loglike - results.null_loglike))
    degrees_of_freedom = int(results.n_params)
    lr_pvalue = None
    if degrees_of_freedom > 0:
        # Chi-square survival probability via the regularized upper incomplete
        # gamma function avoids an additional SciPy dependency.
        lr_pvalue = float(
            torch.special.gammaincc(
                torch.tensor(degrees_of_freedom / 2.0, dtype=torch.float64),
                torch.tensor(likelihood_ratio / 2.0, dtype=torch.float64),
            )
        )
    initial_loglike = results.convergence_status.get("initial_loglike")
    return OrderedDict(
        [
            ("Estimated parameters", results.n_params),
            ("Starting log likelihood", initial_loglike),
            (
                "Log-likelihood improvement",
                results.loglike - initial_loglike if initial_loglike is not None else None,
            ),
            ("Null log likelihood", results.null_loglike),
            ("Final log likelihood", results.loglike),
            ("Likelihood-ratio statistic (null)", likelihood_ratio),
            ("LR degrees of freedom", degrees_of_freedom),
            ("LR p-value", lr_pvalue),
            ("McFadden rho-square", results.rho2),
            ("Adjusted rho-square", results.rho2_bar),
            ("AIC", results.aic),
            ("BIC", results.bic),
        ]
    )


def _inference_section(
    results: object,
    selected_cov: str,
    confidence_level: float,
) -> OrderedDict[str, Any]:
    available = list(results.covariances.keys())
    method = {
        "classic": "Inverse observed information",
        "robust": "Observation-level sandwich",
        "cluster": "Cluster-level sandwich",
    }.get(selected_cov, selected_cov)
    return OrderedDict(
        [
            ("Covariance estimator", selected_cov),
            ("Covariance construction", method),
            ("Available covariance estimators", available),
            ("Confidence level", confidence_level),
            ("Reference distribution", "Asymptotic standard normal"),
            ("Clusters", results.convergence_status.get("n_clusters")),
        ]
    )


def _parameter_table(results: object, cov_type: str, confidence_level: float) -> pd.DataFrame:
    covariance = results.cov_params(cov_type).detach().cpu().numpy()
    standard_errors = np.sqrt(np.clip(np.diag(covariance), a_min=0.0, a_max=None))
    estimates = np.asarray(results.values, dtype=float)
    initial_values = _initial_values(results.model)
    critical = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(results.param_names):
        null_value = _null_value(name)
        standard_error = float(standard_errors[index])
        estimate = float(estimates[index])
        z_value = (estimate - null_value) / standard_error if standard_error > 0 else np.nan
        p_value = erfc(abs(z_value) / sqrt(2.0)) if isfinite(z_value) else np.nan
        # Preserve a stable column order because the same DataFrame feeds HTML,
        # CSV, LaTeX, JSON records, and console output.
        rows.append(
            {
                "Group": _parameter_group(name, results.model),
                "Parameter": name,
                "Start": initial_values.get(name, np.nan),
                "Estimate": estimate,
                "Std. error": standard_error,
                "z-value": z_value,
                "p-value": p_value,
                "H₀ value": null_value,
                "CI lower": estimate - critical * standard_error,
                "CI upper": estimate + critical * standard_error,
                "Status": "Estimated",
            }
        )
    rows.extend(_fixed_parameter_rows(results.model))
    return pd.DataFrame(rows)


def _fixed_parameter_rows(model: object) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    existing: set[str] = set()

    def append(name: str, value: float, group: str) -> None:
        if name in existing:
            return
        existing.add(name)
        rows.append(
            {
                "Group": group,
                "Parameter": name,
                "Start": float(value),
                "Estimate": float(value),
                "Std. error": np.nan,
                "z-value": np.nan,
                "p-value": np.nan,
                "H₀ value": _null_value(name),
                "CI lower": np.nan,
                "CI upper": np.nan,
                "Status": "Fixed",
            }
        )

    spec = getattr(model, "spec", None)
    for parameter in getattr(spec, "parameters", []):
        if parameter.fixed:
            append(parameter.name, parameter.init, "Utility coefficients")
    for nest_name, nest in getattr(model, "nests", {}).items():
        if nest.fixed:
            append(nest.name or f"LAMBDA_{nest_name.upper()}", nest.init, "Nest parameters")
    for coefficient in getattr(model, "random_coefficients", []):
        if coefficient.fixed:
            append(
                coefficient.sigma_name or f"SIGMA_{coefficient.name}",
                coefficient.sigma_init,
                "Random-coefficient scales",
            )
    return rows


def _initial_values(model: object) -> dict[str, float]:
    values: dict[str, float] = {}
    spec = getattr(model, "spec", None)
    for parameter in getattr(spec, "parameters", []):
        values[parameter.name] = float(parameter.init)
    for nest_name, nest in getattr(model, "nests", {}).items():
        values[nest.name or f"LAMBDA_{nest_name.upper()}"] = float(nest.init)
    random_coefficients = list(getattr(model, "random_coefficients", []))
    for coefficient in random_coefficients:
        values[coefficient.sigma_name or f"SIGMA_{coefficient.name}"] = float(coefficient.sigma_init)
    for left_index, left in enumerate(random_coefficients):
        for right in random_coefficients[:left_index]:
            values[f"CHOL_{left.name}__{right.name}"] = 0.0
    return values


def _alternative_table(data: object) -> pd.DataFrame:
    required = ("alt_id", "chosen_row", "availability", "alt_names")
    if not all(hasattr(data, name) for name in required):
        return pd.DataFrame()
    alt_id = data.alt_id.detach().cpu()
    chosen = alt_id[data.chosen_row.detach().cpu()]
    availability = data.availability.detach().cpu()
    rows = []
    for code, name in enumerate(data.alt_names):
        represented = alt_id == code
        available_count = int((represented & availability).sum())
        chosen_count = int((chosen == code).sum())
        rows.append(
            {
                "Alternative": name,
                "Rows": int(represented.sum()),
                "Available": available_count,
                "Chosen": chosen_count,
                "Observed share": chosen_count / max(1, int(chosen.numel())),
            }
        )
    return pd.DataFrame(rows)


def _append_numerical_warnings(
    results: object,
    parameters: pd.DataFrame,
    covariance: np.ndarray,
    warnings: list[str],
) -> None:
    if not np.isfinite(np.asarray(results.values, dtype=float)).all():
        warnings.append("At least one parameter estimate is non-finite.")
    estimated = parameters[parameters["Status"] == "Estimated"]
    if not np.isfinite(estimated["Std. error"].to_numpy(dtype=float)).all():
        warnings.append("At least one estimated standard error is non-finite.")
    if np.linalg.matrix_rank(covariance) < covariance.shape[0]:
        warnings.append("The selected covariance matrix is rank deficient.")


def _append_model_warnings(results: object, parameters: pd.DataFrame, warnings: list[str]) -> None:
    model = results.model
    for row in parameters.itertuples(index=False):
        name = str(row.Parameter)
        estimate = float(row.Estimate)
        if name.upper().startswith("LAMBDA_"):
            lower = float(getattr(model, "lambda_min", 0.0))
            if estimate - lower < 1e-3 or 1.0 - estimate < 1e-3:
                warnings.append(f"{name} is close to its admissible boundary.")
        if name.upper().startswith("SIGMA_"):
            lower = float(getattr(model, "sigma_min", 0.0))
            if estimate - lower < 1e-6:
                warnings.append(f"{name} is close to its lower boundary.")


def _text_formatters(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        column: (lambda value: _format_number(value))
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
    }


def _parameter_text(parameters: pd.DataFrame, confidence_level: float) -> list[str]:
    lines: list[str] = []
    interval_label = f"{100.0 * confidence_level:.0f}% CI"
    for group, rows in parameters.groupby("Group", sort=False):
        if lines:
            lines.append("")
        lines.append(f"[{group}]")
        display = rows[
            [
                "Parameter",
                "Status",
                "Start",
                "Estimate",
                "Std. error",
                "z-value",
                "p-value",
                "H₀ value",
            ]
        ].copy()
        display[interval_label] = [
            "--" if pd.isna(lower) or pd.isna(upper) else f"[{_format_number(lower)}, {_format_number(upper)}]"
            for lower, upper in zip(rows["CI lower"], rows["CI upper"])
        ]
        lines.append(display.to_string(index=False, formatters=_text_formatters(display), na_rep="--"))
    return lines


def _matrix_text(frame: pd.DataFrame) -> str:
    return frame.to_string(formatters=_text_formatters(frame), na_rep="--")


def _format_utility_expression(expression: object) -> str:
    pieces: list[str] = []
    for term in getattr(expression, "terms", []):
        multiplier = float(term.multiplier)
        core = term.parameter.name
        if term.variable is not None:
            core = f"{core} * {term.variable}"
        magnitude = abs(multiplier)
        if magnitude != 1.0:
            core = f"{_format_number(magnitude)} * {core}"
        if not pieces:
            pieces.append(f"-{core}" if multiplier < 0 else core)
        else:
            pieces.append((" - " if multiplier < 0 else " + ") + core)
    return "".join(pieces) if pieces else "0"


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)
