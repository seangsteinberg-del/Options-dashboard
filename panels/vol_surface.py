"""
Volatility Surface Panel v2
============================
Bloomberg-sourced implied vol surface with SABR model fitting,
realized vol overlay, smile analytics, and multiple viz modes.
"""

from dash import html, dcc, dash_table, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from core.pricing import (
    generate_vol_surface, generate_sabr_vol_surface, fit_sabr,
    sabr_vol, realized_vol_close_to_close, generate_price_history,
)
from core.bloomberg import get_vol_surface, get_spot_prices, get_historical_prices, is_connected
from core.theme import (
    COLORS, CARD_STYLE, CARD_HEADER_STYLE, INPUT_STYLE, LABEL_STYLE,
    CHART_TEMPLATE, STAT_BOX_STYLE, make_stat_style,
)


def _param(label, id_, value, min_val, max_val, step, tooltip=""):
    return html.Div([
        html.Label(label, style=LABEL_STYLE, title=tooltip),
        dcc.Input(id=id_, type="number", value=value, min=min_val, max=max_val,
                  step=step, style={**INPUT_STYLE, "marginBottom": "10px"}, debounce=True),
    ], style={"flex": "1", "minWidth": "120px", "marginRight": "12px"})


def layout():
    return html.Div([
        # ── Ticker + Controls ─────────────────────────────────────
        html.Div([
            html.Div([
                html.Span("VOL SURFACE", style={**CARD_HEADER_STYLE, "display": "inline",
                           "borderBottom": "none", "paddingBottom": "0", "marginBottom": "0"}),
                html.Span("  BLOOMBERG" if is_connected() else "  SYNTHETIC", style={
                    "fontSize": "10px", "fontWeight": "700", "letterSpacing": "1px",
                    "color": COLORS["accent_green"] if is_connected() else COLORS["accent_orange"],
                    "marginLeft": "12px",
                }),
            ], style={**CARD_HEADER_STYLE}),

            html.Div([
                html.Div([
                    html.Label("UNDERLYING", style=LABEL_STYLE),
                    dcc.Dropdown(id="vol-ticker", options=[
                        {"label": t, "value": t} for t in
                        ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "JPM", "GS", "XOM", "GLD"]
                    ], value="SPY", clearable=False, style={"fontSize": "12px", "marginBottom": "10px"}),
                ], style={"flex": "1", "minWidth": "130px", "marginRight": "12px"}),

                html.Div([
                    html.Label("VOL MODEL", style=LABEL_STYLE),
                    dcc.Dropdown(id="vol-model", options=[
                        {"label": "Parametric Skew", "value": "parametric"},
                        {"label": "SABR Model", "value": "sabr"},
                    ], value="parametric", clearable=False,
                        style={"fontSize": "12px", "marginBottom": "10px"}),
                ], style={"flex": "1", "minWidth": "150px", "marginRight": "12px"}),

                _param("BASE VOL", "vol-base", 0.20, 0.01, 3.0, 0.01, "ATM implied volatility"),
                _param("SKEW SLOPE", "vol-skew", -0.15, -2.0, 2.0, 0.01, "Put skew steepness"),
                _param("SMILE", "vol-convex", 0.10, 0.0, 2.0, 0.01, "Butterfly / smile curvature"),
                _param("TERM SLOPE", "vol-term", 0.02, -0.5, 0.5, 0.005, "Vol term structure slope"),
            ], style={"display": "flex", "flexWrap": "wrap", "gap": "4px"}),

            # SABR params (shown when SABR selected)
            html.Div(id="vol-sabr-params", children=[
                html.Div("SABR PARAMETERS", style={
                    **LABEL_STYLE, "fontSize": "11px", "color": COLORS["accent_purple"],
                    "marginTop": "12px", "marginBottom": "8px",
                }),
                html.Div([
                    _param("ALPHA", "vol-sabr-alpha", 0.3, 0.01, 3.0, 0.01, "Vol-of-vol base level"),
                    _param("BETA", "vol-sabr-beta", 0.5, 0.0, 1.0, 0.1, "CEV exponent (0=normal, 1=lognormal)"),
                    _param("RHO", "vol-sabr-rho", -0.30, -0.99, 0.99, 0.01, "Spot-vol correlation"),
                    _param("NU", "vol-sabr-nu", 0.40, 0.01, 3.0, 0.01, "Vol-of-vol"),
                ], style={"display": "flex", "flexWrap": "wrap", "gap": "4px"}),
            ]),

            html.Div([
                html.Div([
                    html.Label("VISUALIZATION", style=LABEL_STYLE),
                    dcc.RadioItems(id="vol-viz-mode", options=[
                        {"label": "  3D Surface", "value": "surface"},
                        {"label": "  Heatmap", "value": "heatmap"},
                        {"label": "  Skew Curves", "value": "skew"},
                        {"label": "  Term Structure", "value": "term"},
                        {"label": "  Smile Fit", "value": "smile_fit"},
                    ], value="surface", inline=True,
                        style={"color": COLORS["text_secondary"], "fontSize": "11px",
                               "fontFamily": "'JetBrains Mono', monospace"},
                        inputStyle={"marginRight": "4px"},
                        labelStyle={"marginRight": "18px", "cursor": "pointer"}),
                ], style={"flex": "3"}),
                html.Div([
                    html.Label("COLOR SCHEME", style=LABEL_STYLE),
                    dcc.Dropdown(id="vol-colorscale", options=[
                        {"label": s, "value": s} for s in
                        ["Plasma", "Viridis", "Inferno", "Turbo", "Electric", "thermal", "ice", "RdBu"]
                    ], value="Plasma", clearable=False, style={"fontSize": "11px"}),
                ], style={"flex": "1", "minWidth": "150px"}),
                html.Div([
                    html.Label("STRIKE RANGE ±%", style=LABEL_STYLE),
                    dcc.Slider(id="vol-strike-range", min=5, max=60, step=5, value=30,
                               marks={i: f"{i}%" for i in range(5, 65, 10)},
                               tooltip={"placement": "bottom"}),
                ], style={"flex": "1.5", "minWidth": "200px"}),
            ], style={"display": "flex", "flexWrap": "wrap", "gap": "16px", "marginTop": "14px"}),
        ], style=CARD_STYLE, className="dashboard-card"),

        # ── Stats Row ─────────────────────────────────────────────
        html.Div(id="vol-stats-row", style={
            "display": "flex", "gap": "10px", "marginBottom": "16px", "flexWrap": "wrap",
        }),

        # ── Main Chart ────────────────────────────────────────────
        html.Div([
            dcc.Loading(
                dcc.Graph(id="vol-surface-chart",
                          config={"displayModeBar": True, "scrollZoom": True,
                                  "modeBarButtonsToAdd": ["toImage"]},
                          style={"height": "580px"}),
                type="dot", color=COLORS["accent_cyan"],
            ),
        ], style=CARD_STYLE, className="dashboard-card"),

        # ── Realized Vol vs Implied Vol Chart ─────────────────────
        html.Div([
            html.Div("REALIZED vs IMPLIED VOLATILITY", style=CARD_HEADER_STYLE),
            dcc.Graph(id="vol-rv-iv-chart", style={"height": "320px"},
                      config={"displayModeBar": True}),
        ], style=CARD_STYLE, className="dashboard-card"),

        # ── Term Structure Table ──────────────────────────────────
        html.Div([
            html.Div("VOLATILITY TERM STRUCTURE", style=CARD_HEADER_STYLE),
            html.Div(id="vol-term-table"),
        ], style=CARD_STYLE, className="dashboard-card"),
    ])


def register_callbacks(app):
    @app.callback(
        [Output("vol-surface-chart", "figure"),
         Output("vol-stats-row", "children"),
         Output("vol-term-table", "children"),
         Output("vol-rv-iv-chart", "figure"),
         Output("vol-sabr-params", "style")],
        [Input("vol-ticker", "value"),
         Input("vol-model", "value"),
         Input("vol-base", "value"),
         Input("vol-skew", "value"),
         Input("vol-convex", "value"),
         Input("vol-term", "value"),
         Input("vol-sabr-alpha", "value"),
         Input("vol-sabr-beta", "value"),
         Input("vol-sabr-rho", "value"),
         Input("vol-sabr-nu", "value"),
         Input("vol-viz-mode", "value"),
         Input("vol-colorscale", "value"),
         Input("vol-strike-range", "value")],
    )
    def update_vol_surface(ticker, model, base_vol, skew, convexity, term_slope,
                           sabr_alpha, sabr_beta, sabr_rho, sabr_nu,
                           viz_mode, colorscale, strike_range_pct):
        # Defaults
        base_vol = base_vol or 0.20
        skew = skew if skew is not None else -0.15
        convexity = convexity or 0.10
        term_slope = term_slope if term_slope is not None else 0.02
        pct = (strike_range_pct or 30) / 100.0

        # Show/hide SABR params
        sabr_style = {"display": "block"} if model == "sabr" else {"display": "none"}

        # Get spot from Bloomberg (or fallback)
        spots = get_spot_prices([ticker or "SPY"])
        spot_info = spots.get(ticker or "SPY", {"price": 100})
        S = spot_info["price"]

        # Generate vol surface
        if model == "sabr":
            strikes, expiries, vol_matrix = generate_sabr_vol_surface(
                S=S, alpha=sabr_alpha or 0.3, beta=sabr_beta or 0.5,
                rho_sabr=sabr_rho or -0.3, nu=sabr_nu or 0.4,
                num_strikes=50, num_expiries=25,
                strike_range=(1 - pct, 1 + pct),
            )
        else:
            strikes, expiries, vol_matrix = generate_vol_surface(
                S=S, base_vol=base_vol, skew_slope=skew,
                skew_convexity=convexity, term_slope=term_slope,
                num_strikes=50, num_expiries=25,
                strike_range=(1 - pct, 1 + pct),
            )

        tpl = CHART_TEMPLATE["layout"]
        fig = go.Figure()

        # ── Build Figure ──────────────────────────────────────
        if viz_mode == "surface":
            fig.add_trace(go.Surface(
                x=strikes, y=expiries, z=vol_matrix * 100,
                colorscale=colorscale, opacity=0.93,
                colorbar=dict(title=dict(text="IV %", font=dict(color=COLORS["text_muted"], size=10)),
                              tickfont=dict(color=COLORS["text_muted"], size=9),
                              len=0.6, thickness=12, outlinewidth=0),
                hovertemplate="<b>K:</b> %{x:.1f}<br><b>T:</b> %{y:.2f}y<br><b>IV:</b> %{z:.2f}%<extra></extra>",
                contours=dict(z=dict(show=True, usecolormap=True, project_z=True, width=1)),
                lighting=dict(ambient=0.6, diffuse=0.7, specular=0.3, roughness=0.5),
            ))
            fig.update_layout(scene=dict(
                xaxis=dict(title="Strike", backgroundcolor="rgba(0,0,0,0)",
                           gridcolor=COLORS["border_subtle"], color=COLORS["text_muted"]),
                yaxis=dict(title="Expiry (yrs)", backgroundcolor="rgba(0,0,0,0)",
                           gridcolor=COLORS["border_subtle"], color=COLORS["text_muted"]),
                zaxis=dict(title="IV (%)", backgroundcolor="rgba(0,0,0,0)",
                           gridcolor=COLORS["border_subtle"], color=COLORS["text_muted"]),
                bgcolor="rgba(0,0,0,0)",
                camera=dict(eye=dict(x=1.5, y=-1.5, z=0.85)),
            ))
            title_text = f"Implied Volatility Surface — {ticker} ({model.upper()})"

        elif viz_mode == "heatmap":
            fig.add_trace(go.Heatmap(
                x=strikes, y=expiries, z=vol_matrix * 100, colorscale=colorscale,
                colorbar=dict(title=dict(text="IV %", font=dict(color=COLORS["text_muted"])),
                              tickfont=dict(color=COLORS["text_muted"])),
                hovertemplate="K: %{x:.1f}<br>T: %{y:.2f}y<br>IV: %{z:.2f}%<extra></extra>",
            ))
            fig.add_vline(x=S, line=dict(color=COLORS["accent_orange"], width=1.5, dash="dash"),
                          annotation_text=f"Spot {S:.0f}", annotation_font=dict(color=COLORS["accent_orange"], size=10))
            title_text = f"IV Heatmap — {ticker}"

        elif viz_mode == "skew":
            n_curves = 7
            indices = np.linspace(0, len(expiries) - 1, n_curves, dtype=int)
            colors = [COLORS["accent_cyan"], COLORS["accent_blue"], COLORS["accent_purple"],
                      COLORS["accent_green"], COLORS["accent_orange"], COLORS["accent_pink"], COLORS["accent_red"]]
            for idx, ei in enumerate(indices):
                fig.add_trace(go.Scatter(
                    x=strikes, y=vol_matrix[ei, :] * 100, mode="lines",
                    name=f"{expiries[ei]:.2f}y",
                    line=dict(color=colors[idx % len(colors)], width=2.5),
                ))
            fig.add_vline(x=S, line=dict(color=COLORS["accent_orange"], width=1, dash="dot"))
            title_text = f"Volatility Skew — {ticker}"

        elif viz_mode == "term":
            atm_idx = np.argmin(np.abs(strikes - S))
            offsets = [-8, -4, 0, 4, 8]
            colors = [COLORS["accent_red"], COLORS["accent_orange"], COLORS["accent_cyan"],
                      COLORS["accent_green"], COLORS["accent_purple"]]
            for i, off in enumerate(offsets):
                si = max(0, min(len(strikes) - 1, atm_idx + off))
                fig.add_trace(go.Scatter(
                    x=expiries, y=vol_matrix[:, si] * 100, mode="lines+markers",
                    name=f"K={strikes[si]:.0f}", line=dict(color=colors[i], width=2),
                    marker=dict(size=4),
                ))
            title_text = f"Term Structure — {ticker}"

        elif viz_mode == "smile_fit":
            # Show market smile vs SABR fit for a selected expiry
            mid_exp_idx = len(expiries) // 3
            T_fit = expiries[mid_exp_idx]
            market_vols = vol_matrix[mid_exp_idx, :]
            F = S * np.exp(0.05 * T_fit)

            # Fit SABR to this smile
            params = fit_sabr(strikes, market_vols, F, T_fit, beta=sabr_beta or 0.5)
            sabr_vols = [sabr_vol(F, K, T_fit, params["alpha"], params["beta"],
                                  params["rho"], params["nu"]) for K in strikes]

            fig.add_trace(go.Scatter(
                x=strikes, y=np.array(market_vols) * 100, mode="lines+markers",
                name="Market", line=dict(color=COLORS["accent_cyan"], width=2.5),
                marker=dict(size=5),
            ))
            fig.add_trace(go.Scatter(
                x=strikes, y=np.array(sabr_vols) * 100, mode="lines",
                name=f"SABR Fit (α={params['alpha']:.3f}, ρ={params['rho']:.3f}, ν={params['nu']:.3f})",
                line=dict(color=COLORS["accent_purple"], width=2.5, dash="dash"),
            ))
            fig.add_trace(go.Scatter(
                x=strikes, y=(np.array(market_vols) - np.array(sabr_vols)) * 100 * 10,
                mode="lines", name="Residual (×10)",
                line=dict(color=COLORS["accent_orange"], width=1, dash="dot"),
            ))
            fig.add_vline(x=S, line=dict(color=COLORS["accent_orange"], width=1, dash="dash"))
            title_text = f"Smile Fit — {ticker} T={T_fit:.2f}y"

        fig.update_layout(
            title=dict(text=title_text, font=dict(color=COLORS["text_primary"], size=14,
                       family="JetBrains Mono")),
            paper_bgcolor=tpl["paper_bgcolor"], plot_bgcolor=tpl["plot_bgcolor"],
            font=tpl["font"], margin=dict(l=50, r=20, t=50, b=40),
            hoverlabel=tpl["hoverlabel"],
            legend=dict(font=dict(color=COLORS["text_secondary"], size=10), bgcolor="rgba(0,0,0,0)"),
            xaxis=tpl.get("xaxis", {}), yaxis=tpl.get("yaxis", {}),
        )

        # ── Stats ─────────────────────────────────────────────
        atm_idx = np.argmin(np.abs(strikes - S))
        short_atm = vol_matrix[0, atm_idx] * 100
        long_atm = vol_matrix[-1, atm_idx] * 100
        min_vol = vol_matrix.min() * 100
        max_vol = vol_matrix.max() * 100
        skew_25d = (vol_matrix[len(expiries) // 4, 0] - vol_matrix[len(expiries) // 4, -1]) * 100
        fly = ((vol_matrix[len(expiries) // 4, max(0, atm_idx - 8)] +
                vol_matrix[len(expiries) // 4, min(len(strikes) - 1, atm_idx + 8)]) / 2 -
               vol_matrix[len(expiries) // 4, atm_idx]) * 100

        def stat_box(label, value, fmt, color):
            return html.Div([
                html.Div(f"{value:{fmt}}", className="stat-value", style={"color": color}),
                html.Div(label, className="stat-label"),
            ], style={**make_stat_style(color), "flex": "1"}, className="stat-box")

        stats = [
            stat_box("SPOT", S, ",.2f", COLORS["text_primary"]),
            stat_box("ATM SHORT IV", short_atm, ".2f%", COLORS["accent_cyan"]),
            stat_box("ATM LONG IV", long_atm, ".2f%", COLORS["accent_blue"]),
            stat_box("MIN IV", min_vol, ".2f%", COLORS["accent_green"]),
            stat_box("MAX IV", max_vol, ".2f%", COLORS["accent_red"]),
            stat_box("25D SKEW", skew_25d, "+.2f%", COLORS["accent_purple"]),
            stat_box("BUTTERFLY", fly, "+.2f%", COLORS["accent_orange"]),
            stat_box("TERM SPREAD", long_atm - short_atm, "+.2f%", COLORS["accent_teal"]),
        ]

        # ── Realized vs Implied Vol Chart ─────────────────────
        hist = get_historical_prices(ticker or "SPY", days=252)
        rv_iv_fig = make_subplots(rows=1, cols=1)

        if not hist.empty and "close" in hist.columns:
            prices_arr = hist["close"].values
            rv20 = realized_vol_close_to_close(prices_arr, window=20)
            rv60 = realized_vol_close_to_close(prices_arr, window=60)
            dates = hist.index[-len(rv20):]

            rv_iv_fig.add_trace(go.Scatter(
                x=dates, y=rv20 * 100, mode="lines", name="RV 20d",
                line=dict(color=COLORS["accent_cyan"], width=2),
            ))
            rv_iv_fig.add_trace(go.Scatter(
                x=dates, y=rv60 * 100, mode="lines", name="RV 60d",
                line=dict(color=COLORS["accent_blue"], width=2),
            ))
            # ATM IV line
            rv_iv_fig.add_hline(y=short_atm, line=dict(color=COLORS["accent_orange"], width=1.5, dash="dash"),
                                annotation_text=f"ATM IV {short_atm:.1f}%",
                                annotation_font=dict(color=COLORS["accent_orange"], size=10))

        rv_iv_fig.update_layout(
            paper_bgcolor=tpl["paper_bgcolor"], plot_bgcolor=tpl["plot_bgcolor"],
            font=tpl["font"], margin=dict(l=50, r=20, t=20, b=40),
            hoverlabel=tpl["hoverlabel"], showlegend=True,
            legend=dict(font=dict(color=COLORS["text_secondary"], size=10), bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="rgba(30,42,69,0.5)"),
            yaxis=dict(title="Volatility (%)", gridcolor="rgba(30,42,69,0.5)"),
        )

        # ── Term Structure Table ──────────────────────────────
        term_data = []
        step = max(1, len(expiries) // 10)
        for i in range(0, len(expiries), step):
            T = expiries[i]
            put_idx = max(0, atm_idx - 6)
            call_idx = min(len(strikes) - 1, atm_idx + 6)
            term_data.append({
                "Expiry": f"{T:.3f}y",
                "ATM IV": f"{vol_matrix[i, atm_idx] * 100:.2f}%",
                "25D Put": f"{vol_matrix[i, put_idx] * 100:.2f}%",
                "25D Call": f"{vol_matrix[i, call_idx] * 100:.2f}%",
                "Skew": f"{(vol_matrix[i, put_idx] - vol_matrix[i, call_idx]) * 100:+.2f}%",
                "Butterfly": f"{((vol_matrix[i, put_idx] + vol_matrix[i, call_idx]) / 2 - vol_matrix[i, atm_idx]) * 100:+.2f}%",
            })

        term_table = dash_table.DataTable(
            data=term_data,
            columns=[{"name": c, "id": c} for c in term_data[0].keys()],
            style_header={"backgroundColor": COLORS["bg_secondary"], "color": COLORS["text_secondary"],
                          "fontWeight": "700", "fontSize": "10px", "textTransform": "uppercase",
                          "letterSpacing": "1px", "border": f"1px solid {COLORS['border_subtle']}", "padding": "10px"},
            style_cell={"backgroundColor": COLORS["bg_card"], "color": COLORS["text_primary"],
                        "fontSize": "12px", "fontFamily": "'JetBrains Mono', monospace",
                        "border": f"1px solid {COLORS['border_subtle']}", "padding": "8px 12px", "textAlign": "center"},
            style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": COLORS["bg_secondary"]}],
        )

        return fig, stats, term_table, rv_iv_fig, sabr_style
