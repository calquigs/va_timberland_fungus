-- Macro: Add simplified geometry columns for different zoom levels.
-- Usage: {{ simplify_geom('geom') }} in a model select list.
-- Produces geom_simplified_state (state-level ~1km) and geom_simplified_county (~100m).

{% macro simplify_geom(geom_col) %}
    ST_Simplify({{ geom_col }}, 0.01) as geom_simplified_state,
    ST_Simplify({{ geom_col }}, 0.001) as geom_simplified_county
{% endmacro %}
