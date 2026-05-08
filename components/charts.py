import pandas as pd
import streamlit as st


def _safe_dataframe(data):
    if data is None:
        return pd.DataFrame()

    if isinstance(data, pd.DataFrame):
        return data.copy()

    try:
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()


def render_interactive_chart(
    fig,
    data=None,
    title=None,
    key=None,
    height=None,
    show_data=True,
    show_download=True,
    selection_mode=("points", "box", "lasso"),
):
    """
    Rendert ein Plotly-Diagramm wissenschaftlich nachvollziehbar:
    - interaktiver Plot
    - auswählbare Punkte
    - Datentabelle unter dem Diagramm
    - CSV-Export
    - Anzeige der Selection-Rohdaten
    """

    if title:
        st.markdown(f"### {title}")

    if height is not None:
        try:
            fig.update_layout(height=height)
        except Exception:
            pass

    try:
        fig.update_layout(
            hovermode="closest",
            dragmode="select",
            clickmode="event+select",
        )
        fig.update_traces(
            selected=dict(marker=dict(opacity=1.0)),
            unselected=dict(marker=dict(opacity=0.35)),
        )
    except Exception:
        pass

    chart_key = key or f"chart_{abs(hash(str(title))) % 100000}"

    selection = st.plotly_chart(
        fig,
        width="stretch",
        key=chart_key,
        on_select="rerun",
        selection_mode=list(selection_mode),
        config={
            "displayModeBar": True,
            "responsive": True,
            "scrollZoom": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d"] if "lasso" not in selection_mode else [],
        },
    )

    df = _safe_dataframe(data)

    if show_data and not df.empty:
        with st.expander("Daten hinter dem Diagramm anzeigen", expanded=False):
            st.caption(
                "Diese Tabelle zeigt die Datenbasis, auf der das Diagramm basiert. "
                "Sie kann sortiert, durchsucht und exportiert werden."
            )

            st.dataframe(
                df,
                width="stretch",
                hide_index=True,
            )

            if show_download:
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Daten als CSV herunterladen",
                    data=csv,
                    file_name=f"{chart_key}_daten.csv",
                    mime="text/csv",
                    key=f"{chart_key}_download",
                )

    try:
        selected_points = selection.get("selection", {}).get("points", [])
    except Exception:
        selected_points = []

    if selected_points:
        with st.expander("Ausgewählte Datenpunkte", expanded=True):
            st.caption(
                "Diese Werte stammen direkt aus der Diagramm-Auswahl. "
                "Das hilft, einzelne Balken oder Punkte nachvollziehbar zu prüfen."
            )
            st.json(selected_points)

    return selection


def add_hover_data(fig, hover_template=None):
    """
    Ergänzt ein konsistentes Hover-Verhalten.
    Falls kein eigenes Template übergeben wird, bleibt Plotlys Standard erhalten.
    """
    if hover_template:
        try:
            fig.update_traces(hovertemplate=hover_template)
        except Exception:
            pass

    return fig
