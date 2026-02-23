-- Use custom schema as-is so marts go to "marts" and staging to "staging"
-- (default dbt would produce "staging_marts" when profile schema is "staging")
{% macro generate_schema_name(custom_schema_name, node) %}
    {%- if custom_schema_name is none -%}
        {{ target.schema }}
    {%- else -%}
        {{ custom_schema_name | trim }}
    {%- endif -%}
{% endmacro %}
