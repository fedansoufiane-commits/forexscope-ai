"""Small hand-drawn line-icon set (stroke-based, 24x24) used in place of emoji
across cards/badges/nav. Kept intentionally minimal — geometric shapes, not a
copied icon-font — so the app has its own visual voice instead of a mixed
emoji set."""
from __future__ import annotations

_ICONS = {
    "home": '<path d="M4 12 12 5l8 7"/><path d="M6 10.5V19a1 1 0 0 0 1 1h3v-5h4v5h3a1 1 0 0 0 1-1v-8.5"/>',
    "chart": '<path d="M4 20V10"/><path d="M11 20V4"/><path d="M18 20v-7"/><path d="M3 20h18"/>',
    "pulse": '<path d="M3 12h4l2 7 4-14 2 7h6"/>',
    "compass": '<circle cx="12" cy="12" r="8.5"/><path d="M14.5 9.5 13 13l-3.5 1.5L11 11z"/>',
    "briefcase": '<rect x="3.5" y="7.5" width="17" height="12" rx="1.5"/><path d="M8.5 7.5V6a2 2 0 0 1 2-2h3a2 2 0 0 1 2 2v1.5"/><path d="M3.5 12.5h17"/>',
    "layers": '<path d="M12 4 3.5 8.5 12 13l8.5-4.5z"/><path d="M4.5 12 12 16l7.5-4"/><path d="M4.5 15.5 12 19.5l7.5-4"/>',
    "flask": '<path d="M10 3h4"/><path d="M10 3v6.5L5.5 18a1.5 1.5 0 0 0 1.3 2.3h10.4a1.5 1.5 0 0 0 1.3-2.3L14 9.5V3"/><path d="M8 15h8"/>',
    "newspaper": '<rect x="3.5" y="5.5" width="13" height="14" rx="1"/><path d="M16.5 8.5h3.5a.5.5 0 0 1 .5.5v9a1.5 1.5 0 0 1-1.5 1.5h-3"/><path d="M7 9.5h6"/><path d="M7 13h6"/><path d="M7 16h4"/>',
    "message": '<path d="M4 5.5h16v10a1 1 0 0 1-1 1H9l-4 4v-4H4Z"/>',
    "book": '<path d="M4 5.5A1.5 1.5 0 0 1 5.5 4H12v16H5.5A1.5 1.5 0 0 1 4 18.5Z"/><path d="M20 5.5A1.5 1.5 0 0 0 18.5 4H12v16h6.5a1.5 1.5 0 0 0 1.5-1.5Z"/>',
    "info": '<circle cx="12" cy="12" r="8.5"/><path d="M12 11v5"/><circle cx="12" cy="8" r="0.6" fill="currentColor" stroke="none"/>',
    "download": '<path d="M12 4v11"/><path d="M8 11.5 12 15.5 16 11.5"/><path d="M5 19.5h14"/>',
    "scale": '<path d="M12 3v3"/><path d="M5 20h14"/><path d="M12 6 4.5 9.5"/><path d="M12 6l7.5 3.5"/><path d="M2.5 9.5h5l-2.5 5.5-2.5-5.5Z"/><path d="M16.5 9.5h5L19 15l-2.5-5.5Z"/>',
    "lock": '<rect x="5" y="11" width="14" height="9" rx="1.5"/><path d="M8 11V7.5a4 4 0 0 1 8 0V11"/>',
    "activity": '<circle cx="12" cy="12" r="8.5"/><path d="M8 12h2l1.5 3 2-6 1.5 3h2"/>',
    "trend-up": '<path d="M4 16.5 10 10l4 4 6.5-7"/><path d="M15 6.5h5.5V12"/>',
    "trend-down": '<path d="M4 8.5 10 15l4-4 6.5 7"/><path d="M15 18.5h5.5V13"/>',
    "warning": '<path d="M12 4 21 19H3Z"/><path d="M12 10v4"/><circle cx="12" cy="16.5" r="0.6" fill="currentColor" stroke="none"/>',
    "check": '<circle cx="12" cy="12" r="8.5"/><path d="M8 12.5 11 15.5 16 9"/>',
    "target": '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="0.6" fill="currentColor" stroke="none"/>',
    "grid": '<path d="M4 9h16"/><path d="M4 15h16"/><path d="M9 4v16"/><path d="M15 4v16"/>',
    "link": '<path d="M9.5 13.5 4 8"/><path d="M13 6l3.5-3.5a3 3 0 0 1 4.2 4.2L17 10"/><path d="M11 18l-3.5 3.5a3 3 0 0 1-4.2-4.2L6.5 14"/><path d="M14.5 9.5l1 1"/>',
}


def icon(name: str, size: int = 16, color: str = "currentColor", stroke_width: float = 1.8) -> str:
    body = _ICONS.get(name, _ICONS["info"])
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'xmlns="http://www.w3.org/2000/svg" style="vertical-align:-3px;flex-shrink:0" '
        f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">'
        f'{body}</svg>'
    )
