"""Database connection utilities shared across pages."""

import os

import streamlit as st
from sqlalchemy import create_engine


@st.cache_resource
def get_engine():
    url = os.environ.get(
        "DATABASE_URL",
        "postgresql://va_woods:va_woods_dev@localhost:5432/va_woods",
    )
    return create_engine(url)
