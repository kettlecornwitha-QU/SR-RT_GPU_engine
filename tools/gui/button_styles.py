#!/usr/bin/env python3
from PySide6.QtWidgets import QPushButton

REGULAR_BUTTON_STYLE = (
    "QPushButton {"
    "background-color: #bfbfbf;"
    "border: 1px solid #6f6f6f;"
    "border-top-color: #e7e7e7;"
    "border-bottom-color: #585858;"
    "border-radius: 6px;"
    "padding: 6px 14px;"
    "color: #111111;"
    "}"
    "QPushButton:hover { background-color: #c9c9c9; }"
    "QPushButton:pressed {"
    "background-color: #a9a9a9;"
    "border-top-color: #666666;"
    "border-bottom-color: #dddddd;"
    "}"
)

PRIMARY_BUTTON_STYLE = (
    "QPushButton {"
    "background-color: #2d7df6;"
    "color: white;"
    "border: 1px solid #1d5fc5;"
    "border-top-color: #6fa7ff;"
    "border-bottom-color: #174c9f;"
    "border-radius: 6px;"
    "padding: 6px 14px;"
    "}"
    "QPushButton:hover { background-color: #3b89ff; }"
    "QPushButton:pressed {"
    "background-color: #1f6ede;"
    "border-top-color: #1757b8;"
    "border-bottom-color: #8ab8ff;"
    "}"
)


def style_button(button: QPushButton, *, primary: bool = False) -> None:
    button.setStyleSheet(PRIMARY_BUTTON_STYLE if primary else REGULAR_BUTTON_STYLE)
