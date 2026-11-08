# SPDX-License-Identifier: BSD-3-Clause

function(_ernie_schema_write path variable)
    file(WRITE "${path}.tmp" "${${variable}}")
    file(COPY_FILE "${path}.tmp" "${path}" ONLY_IF_DIFFERENT)
    file(REMOVE "${path}.tmp")
endfunction()

set(_ernie_schema_units
    meters
    millimeters
    degrees
    radians
    kelvin
    seconds
    nits
    lux
    watts
    hertz
)
set(_ernie_schema_metadata_keys
    "tracked|texturable|doc[ \t]*=|unit[ \t]*=|allowed[ \t]*=|enum[ \t]*=|group[ \t]*=|displayName[ \t]*=|enabledBy[ \t]*=|min[ \t]*=|max[ \t]*="
)
set(_ernie_schema_number "-?[0-9]+(\\.[0-9]*)?([eE][-+]?[0-9]+)?")

function(_ernie_schema_enum_name token out)
    string(
        REPLACE "-"
        " minus "
        words
        "${token}"
    )
    string(
        REPLACE "+"
        " plus "
        words
        "${words}"
    )
    string(
        REGEX REPLACE "[ \t]+"
        ";"
        words
        "${words}"
    )
    set(name "")
    foreach(word IN LISTS words)
        if(NOT word STREQUAL "")
            string(SUBSTRING "${word}" 0 1 first)
            string(SUBSTRING "${word}" 1 -1 others)
            string(TOUPPER "${first}" first)
            string(APPEND name "${first}${others}")
        endif()
    endforeach()
    set(${out} "${name}" PARENT_SCOPE)
endfunction()

function(_ernie_schema_capitalize text out)
    string(SUBSTRING "${text}" 0 1 first)
    string(SUBSTRING "${text}" 1 -1 others)
    string(TOUPPER "${first}" first)
    set(${out} "${first}${others}" PARENT_SCOPE)
endfunction()

set(_ernie_schema_namespaces
    inputs
    outputs
    primvars
    info
    xformOp
    omni
    ernie
)

function(_ernie_schema_field attribute out)
    string(
        REPLACE ":"
        ";"
        parts
        "${attribute}"
    )
    list(LENGTH parts count)
    while(count GREATER 1)
        list(GET parts 0 first)
        if(NOT first IN_LIST _ernie_schema_namespaces)
            break()
        endif()
        list(REMOVE_AT parts 0)
        math(EXPR count "${count} - 1")
    endwhile()
    list(POP_FRONT parts field)
    foreach(part IN LISTS parts)
        _ernie_schema_capitalize("${part}" part)
        string(APPEND field "${part}")
    endforeach()
    set(${out} "${field}" PARENT_SCOPE)
endfunction()

function(
    _ernie_schema_metadata
    text
    where
    attribute
    out_flags
    out_doc
    out_unit
    out_allowed
    out_enum
    out_group
    out_display
    out_enabled
    out_min
    out_max
    out_applies
    out_auto
)
    set(flags "")
    set(doc "")
    set(unit None)
    set(allowed "")
    set(enum "")
    set(group "")
    set(display "")
    set(enabled "")
    set(min 0)
    set(max 0)
    set(applies "")
    set(auto "")
    string(STRIP "${text}" text)
    while(NOT text STREQUAL "")
        if(attribute AND text MATCHES "^(tracked|texturable)[ \t]*(,(.*))?$")
            list(APPEND flags ${CMAKE_MATCH_1})
            set(text "${CMAKE_MATCH_3}")
        elseif(text MATCHES "^doc[ \t]*=[ \t]*\"([^\"\\\\]*)\"[ \t]*(,(.*))?$")
            set(doc "${CMAKE_MATCH_1}")
            set(text "${CMAKE_MATCH_3}")
        elseif(
            attribute
            AND text
                MATCHES
                "^allowed[ \t]*=[ \t]*<([A-Za-z0-9_, \t\"+-]*)>[ \t]*(,(.*))?$"
        )
            set(rest "${CMAKE_MATCH_3}")
            string(
                REPLACE ","
                ";"
                values
                "${CMAKE_MATCH_1}"
            )
            set(allowed "")
            foreach(value IN LISTS values)
                string(STRIP "${value}" value)
                if(value STREQUAL "")
                    continue()
                endif()
                if(value MATCHES "^\"([^\"]+)\"$")
                    set(value "${CMAKE_MATCH_1}")
                elseif(NOT value MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
                    message(FATAL_ERROR "${where}: ${value} isn't a token")
                endif()
                list(APPEND allowed "${value}")
            endforeach()
            set(text "${rest}")
        elseif(
            NOT attribute
            AND text
                MATCHES
                "^(appliesTo|autoApplyTo)[ \t]*=[ \t]*<([A-Za-z0-9_, \t]*)>[ \t]*(,(.*))?$"
        )
            set(key "${CMAKE_MATCH_1}")
            set(rest "${CMAKE_MATCH_4}")
            string(
                REPLACE ","
                ";"
                values
                "${CMAKE_MATCH_2}"
            )
            set(names "")
            foreach(value IN LISTS values)
                string(STRIP "${value}" value)
                if(NOT value STREQUAL "")
                    list(APPEND names "${value}")
                endif()
            endforeach()
            if(key STREQUAL "appliesTo")
                set(applies "${names}")
            else()
                set(auto "${names}")
            endif()
            set(text "${rest}")
        elseif(attribute AND text MATCHES "^group[ \t]*=[ \t]*\"([^\"\\\\]*)\"[ \t]*(,(.*))?$")
            set(group "${CMAKE_MATCH_1}")
            set(text "${CMAKE_MATCH_3}")
        elseif(
            attribute
            AND text
                MATCHES
                "^displayName[ \t]*=[ \t]*\"([^\"\\\\]*)\"[ \t]*(,(.*))?$"
        )
            set(display "${CMAKE_MATCH_1}")
            set(text "${CMAKE_MATCH_3}")
        elseif(
            attribute
            AND text
                MATCHES
                "^enabledBy[ \t]*=[ \t]*\"([A-Za-z_][A-Za-z0-9_]*)\"[ \t]*(,(.*))?$"
        )
            set(enabled "${CMAKE_MATCH_1}")
            set(text "${CMAKE_MATCH_3}")
        elseif(
            attribute
            AND text
                MATCHES
                "^(min|max)[ \t]*=[ \t]*(${_ernie_schema_number})[ \t]*(,(.*))?$"
        )
            set(${CMAKE_MATCH_1} "${CMAKE_MATCH_2}")
            set(text "${CMAKE_MATCH_6}")
        elseif(attribute AND text MATCHES "^enum[ \t]*=[ \t]*([A-Z][A-Za-z0-9]*)[ \t]*(,(.*))?$")
            set(enum "${CMAKE_MATCH_1}")
            set(text "${CMAKE_MATCH_3}")
        elseif(attribute AND text MATCHES "^unit[ \t]*=[ \t]*([a-z]+)[ \t]*(,(.*))?$")
            if(NOT CMAKE_MATCH_1 IN_LIST _ernie_schema_units)
                list(JOIN _ernie_schema_units ", " units)
                message(FATAL_ERROR "${where}: ${CMAKE_MATCH_1} isn't a unit; units are ${units}")
            endif()
            string(SUBSTRING "${CMAKE_MATCH_1}" 0 1 first)
            string(SUBSTRING "${CMAKE_MATCH_1}" 1 -1 others)
            string(TOUPPER "${first}" first)
            set(unit "${first}${others}")
            set(text "${CMAKE_MATCH_3}")
        else()
            message(
                FATAL_ERROR
                "${where}: expected tracked, texturable, doc, unit, allowed, enum, group, displayName, enabledBy, min, max, appliesTo or autoApplyTo, not ${text}"
            )
        endif()
        string(STRIP "${text}" text)
    endwhile()
    set(${out_flags} "${flags}" PARENT_SCOPE)
    set(${out_doc} "${doc}" PARENT_SCOPE)
    set(${out_unit} "${unit}" PARENT_SCOPE)
    set(${out_allowed} "${allowed}" PARENT_SCOPE)
    set(${out_enum} "${enum}" PARENT_SCOPE)
    set(${out_group} "${group}" PARENT_SCOPE)
    set(${out_display} "${display}" PARENT_SCOPE)
    set(${out_enabled} "${enabled}" PARENT_SCOPE)
    set(${out_min} "${min}" PARENT_SCOPE)
    set(${out_max} "${max}" PARENT_SCOPE)
    set(${out_applies} "${applies}" PARENT_SCOPE)
    set(${out_auto} "${auto}" PARENT_SCOPE)
endfunction()

function(_ernie_schema_texturable line where out)
    if(NOT line MATCHES "^texturable[ \t]+([a-z0-9_]+)[ \t]+([A-Za-z_][A-Za-z0-9_:]*)(.*)$")
        message(FATAL_ERROR "${where}: expected an attribute after texturable")
    endif()
    set(type "${CMAKE_MATCH_1}")
    set(attribute "${CMAKE_MATCH_2}")
    set(rest "${CMAKE_MATCH_3}")
    if(NOT type MATCHES "^(float|float3|color3f|normal3f)$")
        message(FATAL_ERROR "${where}: only float, float3, color3f and normal3f are texturable")
    endif()
    set(metadata "")
    if(rest MATCHES "^(.*)\\(([ \t]*(${_ernie_schema_metadata_keys}).*)\\)[ \t]*$")
        set(rest "${CMAKE_MATCH_1}")
        set(metadata "${CMAKE_MATCH_2}")
    endif()
    set(group "")
    if(metadata MATCHES "(group[ \t]*=[ \t]*\"[^\"]*\")")
        set(group "${CMAKE_MATCH_1}, ")
    endif()
    if(metadata MATCHES "(enabledBy[ \t]*=[ \t]*\"[^\"]*\")")
        string(APPEND group "${CMAKE_MATCH_1}, ")
    endif()
    _ernie_schema_field("${attribute}" field)
    set(channel rgb)
    if(type STREQUAL "float")
        set(channel r)
    endif()
    if(metadata STREQUAL "")
        set(value "${type} ${attribute}${rest} (texturable)")
    else()
        set(value "${type} ${attribute}${rest} (texturable, ${metadata})")
    endif()
    set(${out}
        "${value}"
        "asset ${attribute}:texture (tracked, ${group}doc = \"A texture whose channels replace ${field}.\")"
        "token ${attribute}:textureChannel = ${channel} (allowed = <rgb, r, g, b, a>, enum = TextureChannel, ${group}doc = \"Which channels of the texture replace ${field}.\")"
        "float ${attribute}:textureInfluence = 1 (min = 0, max = 1, ${group}doc = \"How much the texture replaces ${field}, from not at all at 0 to fully at 1.\")"
        PARENT_SCOPE
    )
endfunction()

function(_ernie_schema_scalar element text where out)
    string(STRIP "${text}" text)
    if(element STREQUAL "int")
        if(NOT text MATCHES "^-?[0-9]+$")
            message(FATAL_ERROR "${where}: ${text} isn't an int")
        endif()
    elseif(element STREQUAL "uint")
        if(NOT text MATCHES "^[0-9]+$")
            message(FATAL_ERROR "${where}: ${text} isn't a uint")
        endif()
        string(APPEND text "u")
    elseif(text MATCHES "^(-?)inf$")
        set(text "${CMAKE_MATCH_1}std::numeric_limits<${element}>::infinity()")
    else()
        if(NOT text MATCHES "^-?([0-9]+\\.?[0-9]*|\\.[0-9]+)([eE][-+]?[0-9]+)?$")
            message(FATAL_ERROR "${where}: ${text} isn't a number")
        endif()
        if(NOT text MATCHES "[.eE]")
            string(APPEND text ".0")
        endif()
        if(element STREQUAL "float")
            string(APPEND text "f")
        endif()
    endif()
    set(${out} "${text}" PARENT_SCOPE)
endfunction()

function(_ernie_schema_types type where gpu cpp)
    if(type STREQUAL "bool")
        set(g uint)
        set(c bool)
    elseif(type STREQUAL "int")
        set(g int)
        set(c int32_t)
    elseif(type STREQUAL "uint")
        set(g uint)
        set(c uint32_t)
    elseif(type STREQUAL "float")
        set(g float)
        set(c float)
    elseif(type STREQUAL "float2")
        set(g Float2)
        set(c Float2)
    elseif(type STREQUAL "uint2")
        set(g Uint2)
        set(c Uint2)
    elseif(type MATCHES "^(float3|color3f|point3f|normal3f|vector3f)$")
        set(g Float3)
        set(c Float3)
    elseif(type MATCHES "^(float4|color4f)$")
        set(g Float4)
        set(c Float4)
    elseif(type STREQUAL "double3")
        set(g Double3)
        set(c Double3)
    elseif(type STREQUAL "quatd")
        set(g Quatd)
        set(c Quatd)
    elseif(type STREQUAL "uint64")
        set(g uint64_t)
        set(c uint64_t)
    elseif(type STREQUAL "transform")
        set(g WorldTransform)
        set(c WorldTransform)
    elseif(type MATCHES "^enum_(.+)$")
        set(g uint)
        set(c "${CMAKE_MATCH_1}")
    elseif(type MATCHES "^(token|asset)$")
        set(g uint)
        set(c "std::string")
    elseif(type STREQUAL "int_array")
        set(g ArrayRef)
        set(c "std::vector<int32_t>")
    elseif(type STREQUAL "int64_array")
        set(g ArrayRef)
        set(c "std::vector<int64_t>")
    elseif(type MATCHES "^(prim|token)_array$")
        set(g ArrayRef)
        set(c "std::vector<std::string>")
    elseif(type STREQUAL "primvars")
        set(g ArrayRef)
        set(c "std::vector<Primvar>")
    elseif(type STREQUAL "inputs")
        set(g "")
        set(c "std::vector<MaterialInput>")
    elseif(type MATCHES "^(float3|color3f|point3f|normal3f|vector3f)_array$")
        set(g ArrayRef)
        set(c "std::vector<Float3>")
    elseif(type STREQUAL "float_array")
        set(g ArrayRef)
        set(c "std::vector<float>")
    elseif(type STREQUAL "float2_array")
        set(g ArrayRef)
        set(c "std::vector<Float2>")
    elseif(type MATCHES "^(float4|color4f|quatf|quath)_array$")
        set(g ArrayRef)
        set(c "std::vector<Float4>")
    else()
        string(
            REPLACE "_array"
            "[]"
            type
            "${type}"
        )
        message(FATAL_ERROR "${where}: ${type} isn't a column type")
    endif()
    set(${gpu} "${g}" PARENT_SCOPE)
    set(${cpp} "${c}" PARENT_SCOPE)
endfunction()

function(_ernie_schema_column type out_type out_role)
    set(role None)
    if(type STREQUAL "color4f")
        set(role Color)
    elseif(type MATCHES "^(color3f|point3f|normal3f|vector3f)(_array)?$")
        string(
            REGEX REPLACE "3f.*$"
            ""
            role
            "${type}"
        )
        string(SUBSTRING "${role}" 0 1 first)
        string(SUBSTRING "${role}" 1 -1 others)
        string(TOUPPER "${first}" first)
        set(role "${first}${others}")
    endif()
    if(type MATCHES "^(float3|color3f|point3f|normal3f|vector3f)_array$")
        set(column Float3Array)
    elseif(type MATCHES "^(float4|color4f|quatf|quath)_array$")
        set(column Float4Array)
    elseif(type STREQUAL "int64_array")
        set(column Int64Array)
    elseif(type STREQUAL "float_array")
        set(column FloatArray)
    elseif(type STREQUAL "float2_array")
        set(column Float2Array)
    elseif(type MATCHES "^(float3|color3f|point3f|normal3f|vector3f)$")
        set(column Float3)
    elseif(type MATCHES "^(float4|color4f)$")
        set(column Float4)
    elseif(type MATCHES "^enum_")
        set(column Enum)
    elseif(type STREQUAL "token")
        set(column Token)
    elseif(type STREQUAL "asset")
        set(column Asset)
    elseif(type MATCHES "^(int_array|primvars)$")
        set(column IntArray)
    elseif(type STREQUAL "prim_array")
        set(column PrimArray)
    elseif(type STREQUAL "token_array")
        set(column TokenArray)
    elseif(type STREQUAL "double3")
        set(column Double3)
    elseif(type STREQUAL "quatd")
        set(column Quatd)
    elseif(type STREQUAL "transform")
        set(column WorldTransform)
    else()
        string(SUBSTRING "${type}" 0 1 first)
        string(SUBSTRING "${type}" 1 -1 others)
        string(TOUPPER "${first}" first)
        set(column "${first}${others}")
    endif()
    set(${out_type} "${column}" PARENT_SCOPE)
    set(${out_role} "${role}" PARENT_SCOPE)
endfunction()

function(_ernie_schema_value type value where out)
    if(value STREQUAL "")
        set(${out} "{}" PARENT_SCOPE)
        return()
    endif()
    if(type MATCHES "(_array|^primvars|^inputs)$")
        message(FATAL_ERROR "${where}: arrays take no fallback")
    endif()
    if(type MATCHES "^enum_(.+)$")
        set(enum "${CMAKE_MATCH_1}")
        if(value MATCHES "^\"([^\"]+)\"$")
            set(value "${CMAKE_MATCH_1}")
        endif()
        if(NOT value IN_LIST _ernie_schema_allowed_${enum})
            list(JOIN _ernie_schema_allowed_${enum} ", " allowed)
            message(FATAL_ERROR "${where}: ${value} isn't one of ${allowed}")
        endif()
        _ernie_schema_enum_name("${value}" value)
        set(${out} "${enum}::${value}" PARENT_SCOPE)
        return()
    endif()
    if(type STREQUAL "asset")
        if(NOT value MATCHES "^@([^@\"]*)@$")
            message(FATAL_ERROR "${where}: ${value} isn't an asset path like @file.png@")
        endif()
        set(${out} "\"${CMAKE_MATCH_1}\"" PARENT_SCOPE)
        return()
    endif()
    if(type STREQUAL "token")
        if(NOT value MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
            message(FATAL_ERROR "${where}: ${value} isn't a token")
        endif()
        set(${out} "\"${value}\"" PARENT_SCOPE)
        return()
    endif()
    if(type STREQUAL "bool")
        if(NOT value MATCHES "^(true|false)$")
            message(FATAL_ERROR "${where}: ${value} isn't a bool")
        endif()
        set(${out} "${value}" PARENT_SCOPE)
        return()
    endif()
    if(type MATCHES "^(int|uint|float)$")
        _ernie_schema_scalar(${type} "${value}" "${where}" text)
        set(${out} "${text}" PARENT_SCOPE)
        return()
    endif()
    if(type STREQUAL "float2")
        set(element float)
        set(lanes 2)
    elseif(type STREQUAL "uint2")
        set(element uint)
        set(lanes 2)
    elseif(type STREQUAL "double3")
        set(element double)
        set(lanes 3)
    elseif(type STREQUAL "quatd")
        set(element double)
        set(lanes 4)
    elseif(type MATCHES "^(float4|color4f)$")
        set(element float)
        set(lanes 4)
    else()
        set(element float)
        set(lanes 3)
    endif()
    if(NOT value MATCHES "^\\((.*)\\)$")
        message(FATAL_ERROR "${where}: ${value} isn't a tuple of ${lanes}")
    endif()
    string(
        REPLACE ","
        ";"
        parts
        "${CMAKE_MATCH_1}"
    )
    list(LENGTH parts count)
    if(NOT count EQUAL lanes)
        message(FATAL_ERROR "${where}: ${value} isn't a tuple of ${lanes}")
    endif()
    set(items "")
    foreach(part IN LISTS parts)
        _ernie_schema_scalar(${element} "${part}" "${where}" item)
        list(APPEND items "${item}")
    endforeach()
    list(JOIN items ", " joined)
    set(${out} "{${joined}}" PARENT_SCOPE)
endfunction()

function(
    _ernie_schema_python_types
    type
    out_value
    out_input
    out_many
    out_many_input
)
    set(value "")
    set(input "")
    set(many "")
    set(many_input "")
    if(type STREQUAL "bool")
        set(value bool)
        set(input bool)
        set(many "NDArray[np.bool_]")
    elseif(type MATCHES "^u?int$")
        set(value int)
        set(input int)
        set(many "NDArray[np.${type}32]")
    elseif(type STREQUAL "float")
        set(value float)
        set(input float)
        set(many "NDArray[np.float32]")
    elseif(type STREQUAL "uint64")
        set(value int)
        set(input int)
        set(many "NDArray[np.uint64]")
    elseif(type STREQUAL "transform")
        set(value "NDArray[np.float32]")
        set(many "NDArray[np.float32]")
    elseif(type STREQUAL "float2")
        set(value "tuple[float, float]")
        set(many "NDArray[np.float32]")
    elseif(type STREQUAL "uint2")
        set(value "tuple[int, int]")
        set(many "NDArray[np.uint32]")
    elseif(type MATCHES "^(float3|color3f|point3f|normal3f|vector3f)$")
        set(value "tuple[float, float, float]")
        set(many "NDArray[np.float32]")
    elseif(type MATCHES "^(float4|color4f)$")
        set(value "tuple[float, float, float, float]")
        set(many "NDArray[np.float32]")
    elseif(type STREQUAL "double3")
        set(value "tuple[float, float, float]")
        set(many "NDArray[np.float64]")
    elseif(type STREQUAL "quatd")
        set(value "tuple[float, float, float, float]")
        set(many "NDArray[np.float64]")
    elseif(type MATCHES "^enum_(.+)$")
        set(quoted "")
        foreach(token IN LISTS _ernie_schema_allowed_${CMAKE_MATCH_1})
            list(APPEND quoted "\"${token}\"")
        endforeach()
        list(JOIN quoted ", " quoted)
        set(value "Literal[${quoted}]")
        set(input "${value}")
        set(many "list[${value}]")
        set(many_input "${value} | Sequence[${value}]")
    elseif(type MATCHES "^(token|asset)$")
        set(value str)
        set(input str)
        set(many "list[str]")
        set(many_input "str | Sequence[str]")
    elseif(type STREQUAL "int_array")
        set(value "NDArray[np.int32]")
    elseif(type STREQUAL "int64_array")
        set(value "NDArray[np.int64]")
    elseif(type MATCHES "_array$" AND NOT type MATCHES "^(prim|token)_array$")
        set(value "NDArray[np.float32]")
    elseif(type STREQUAL "prim_array")
        set(value "list[str]")
        set(input Targets)
    elseif(type STREQUAL "token_array")
        set(value "list[str]")
        set(input "Sequence[str]")
    elseif(type STREQUAL "primvars")
        set(value "list[Primvar]")
        set(input "Sequence[Primvar]")
    endif()
    if(NOT input)
        set(input ArrayLike)
    endif()
    if(NOT many)
        set(many "list[${value}]")
        set(many_input "${input}")
    elseif(NOT many_input)
        set(many_input ArrayLike)
    endif()
    set(${out_value} "${value}" PARENT_SCOPE)
    set(${out_input} "${input}" PARENT_SCOPE)
    set(${out_many} "${many}" PARENT_SCOPE)
    set(${out_many_input} "${many_input}" PARENT_SCOPE)
endfunction()

function(_ernie_schema_python_name field out)
    string(
        REGEX REPLACE "([A-Z])([A-Z][a-z])"
        "\\1_\\2"
        field
        "${field}"
    )
    string(
        REGEX REPLACE "([a-z0-9])([A-Z])"
        "\\1_\\2"
        field
        "${field}"
    )
    string(TOLOWER "${field}" field)
    set(${out} "${field}" PARENT_SCOPE)
endfunction()

function(_ernie_schema_is_a schema bases out)
    set(found FALSE)
    set(level ${schema})
    while(NOT level STREQUAL "None")
        if(level IN_LIST bases)
            set(found TRUE)
            break()
        endif()
        set(level ${base_${level}})
    endwhile()
    set(${out} ${found} PARENT_SCOPE)
endfunction()

function(_ernie_schema_python name out_prim out_selection)
    set(parents "")
    if(NOT base_${name} STREQUAL "None")
        list(APPEND parents ${base_${name}})
    endif()
    list(APPEND parents ${apis_${name}})
    list(TRANSFORM parents APPEND "Selection" OUTPUT_VARIABLE selection_parents)
    if(NOT parents)
        set(parents SchemaPrim)
        set(selection_parents SchemaSelection)
        if(multiple_${name})
            set(parents InstancePrim)
        endif()
    endif()
    list(JOIN parents ", " parents)
    list(JOIN selection_parents ", " selection_parents)
    set(api False)
    if(kind_${name} STREQUAL "Api")
        set(api True)
    endif()
    set(prim "class ${name}(${parents}):\n")
    if(doc_${name})
        string(APPEND prim "    \"${doc_${name}}\"\n\n")
    endif()
    string(
        APPEND prim
        "    schema: ClassVar[str] = \"${name}\"\n    api: ClassVar[bool] = ${api}\n    __slots__ = ()\n"
    )
    set(selection
        "class ${name}Selection(${selection_parents}):\n    schema: ClassVar[str] = \"${name}\"\n    __slots__ = ()\n"
    )
    set(parameters "")
    set(values "")
    set(fields "")
    foreach(field IN LISTS desc)
        if(desc_type_${field} STREQUAL "inputs")
            continue()
        endif()
        _ernie_schema_python_name(${field} python)
        _ernie_schema_python_types(${desc_type_${field}} value input many many_input)
        string(APPEND parameters "        ${python}: ${input} | None = None,\n")
        string(APPEND fields "        \"${python}\": \"${field}\",\n")
        if(desc_type_${field} STREQUAL "prim_array")
            string(APPEND values "                    ${field}=_targets(${python}),\n")
        else()
            string(APPEND values "                    ${field}=${python},\n")
        endif()
    endforeach()
    set(api_values "")
    set(instance_properties "")
    foreach(api IN LISTS optional)
        set(member ${member_${name}_${api}})
        _ernie_schema_python_name(${member} python)
        if(NOT multiple_${api})
            string(APPEND parameters "        ${python}: Mapping[str, Any] | None = None,\n")
            string(APPEND api_values "                    (${api}, ${python}),\n")
        elseif(single_${name}_${api})
            set(field ${single_${name}_${api}})
            _ernie_schema_python_types(${type_${api}_${field}} value input many many_input)
            string(APPEND parameters "        ${python}: Mapping[str, ${input}] | None = None,\n")
            string(APPEND values "                    ${member}=${python},\n")
            string(
                APPEND instance_properties
                "\n    @property\n    def ${python}(self) -> dict[str, ${value}]:\n"
                "        \"${doc_${api}}\"\n"
                "        return self.prim.instance_values(\"${api}\", \"${field}\")\n"
                "\n    @${python}.setter\n    def ${python}(self, value: Mapping[str, ${input}]) -> None:\n"
                "        self.prim.set_instances(\"${api}\", \"${field}\", value)\n"
            )
        else()
            string(
                APPEND parameters
                "        ${python}: Mapping[str, Mapping[str, Any]] | None = None,\n"
            )
            string(APPEND values "                    ${member}=_instances(${api}, ${python}),\n")
        endif()
    endforeach()
    foreach(entry IN LISTS members)
        string(
            REPLACE ":"
            ";"
            parts
            "${entry}"
        )
        list(GET parts 1 api)
        list(GET parts 2 instance)
        _ernie_schema_python_name(${instance} python)
        string(APPEND parameters "        ${python}: Mapping[str, Any] | None = None,\n")
        string(APPEND values "                    ${instance}=_fields(${api}, ${python}),\n")
    endforeach()
    string(APPEND prim "${instance_properties}")
    if(parameters)
        set(parameters "        *,\n${parameters}")
    endif()
    if(kind_${name} STREQUAL "Concrete")
        set(apis_argument "")
        if(api_values)
            set(apis_argument "                apis=_apis(\n${api_values}                ),\n")
        endif()
        string(
            APPEND prim
            "\n    @classmethod\n    def define(\n        cls,\n        scene: Scene,\n        path: str,\n"
            "${parameters}    ) -> ${name}:\n"
            "        return cls(\n            scene.define(\n                path,\n                \"${name}\",\n"
            "                _given(\n${values}                ),\n${apis_argument}            )\n        )\n"
        )
    endif()
    if(kind_${name} STREQUAL "Api" AND multiple_${name})
        string(
            APPEND prim
            "\n    _fields: ClassVar[dict[str, str]] = {\n${fields}    }\n"
            "\n    @classmethod\n    def apply(\n        cls,\n        prims: Prims,\n        instance: str,\n"
            "${parameters}    ) -> None:\n"
            "        _apply(\n            prims,\n            \"${name}\",\n"
            "            _given(\n${values}            ),\n            instance,\n        )\n"
            "\n    @classmethod\n    def unapply(cls, prims: Prims, instance: str | None = None) -> None:\n"
            "        _unapply(prims, \"${name}\", instance)\n"
            "\n    @classmethod\n    def select(cls, scene: Scene, pattern: str, instance: str) -> ${name}Selection:\n"
            "        return ${name}Selection(scene.select(pattern, \"${name}\", instance=instance))\n"
        )
    elseif(kind_${name} STREQUAL "Api")
        string(
            APPEND prim
            "\n    _fields: ClassVar[dict[str, str]] = {\n${fields}    }\n"
            "\n    @classmethod\n    def apply(\n        cls,\n        prims: Prims,\n"
            "${parameters}    ) -> None:\n"
            "        _apply(\n            prims,\n            \"${name}\",\n"
            "            _given(\n${values}            ),\n        )\n"
            "\n    @classmethod\n    def unapply(cls, prims: Prims) -> None:\n"
            "        _unapply(prims, \"${name}\")\n"
        )
    endif()
    if(NOT multiple_${name})
        string(
            APPEND prim
            "\n    @classmethod\n    def select(cls, scene: Scene, pattern: str) -> ${name}Selection:\n"
            "        return ${name}Selection(scene.select(pattern, \"${name}\"))\n"
        )
    endif()
    foreach(field IN LISTS columns)
        if(multiple_${name} AND field MATCHES "^(instance|next)$")
            continue()
        endif()
        _ernie_schema_python_name(${field} python)
        _ernie_schema_python_types(${type_${name}_${field}} value input many many_input)
        set(doc "${doc_${name}_${field}}")
        if(NOT unit_${name}_${field} STREQUAL "None")
            string(TOLOWER "${unit_${name}_${field}}" unit)
            string(APPEND doc " In ${unit}.")
        endif()
        set(argument value)
        if(type_${name}_${field} STREQUAL "prim_array")
            set(argument "_targets(value)")
        endif()
        foreach(target prim selection)
            if(target STREQUAL "prim")
                set(get_type "${value}")
                set(set_type "${input}")
            else()
                set(get_type "${many}")
                set(set_type "${many_input}")
            endif()
            set(read "self.${target}[\"${field}\"]")
            set(write "self.${target}[\"${field}\"] = ${argument}")
            if(multiple_${name} AND target STREQUAL "prim")
                set(read "self.prim.get_instance(self.schema, self.instance, \"${field}\")")
                set(write
                    "self.prim.set_instance(self.schema, self.instance, \"${field}\", ${argument})"
                )
            endif()
            string(
                APPEND ${target}
                "\n    @property\n    def ${python}(self) -> ${get_type}:\n        \"${doc}\"\n"
                "        return ${read}\n"
            )
            if(NOT "internal" IN_LIST flags_${name}_${field})
                string(
                    APPEND ${target}
                    "\n    @${python}.setter\n    def ${python}(self, value: ${set_type}) -> None:\n"
                    "        ${write}\n"
                )
            endif()
        endforeach()
    endforeach()
    set(${out_prim} "${prim}" PARENT_SCOPE)
    set(${out_selection} "${selection}" PARENT_SCOPE)
endfunction()

function(ernie_generate_schemas directory)
    cmake_parse_arguments(
        PARSE_ARGV 1
        ARG
        ""
        "PYTHON;PYTHON_DEFINES;DEFINE_HEADER;DEFINE_SOURCE;TYPED_HEADER"
        ""
    )
    set(schemas "")
    set(tokens_text "")
    set(families "")
    foreach(input IN LISTS ARG_UNPARSED_ARGUMENTS)
        set_property(
            DIRECTORY
            APPEND
            PROPERTY
                CMAKE_CONFIGURE_DEPENDS
                    "${input}"
        )
        get_filename_component(file "${input}" NAME)
        get_filename_component(family "${input}" NAME_WE)
        list(APPEND families ${family})
        set(source_${family} "${file}")
        set(gpu_${family} "")
        set(desc_${family} "")
        file(READ "${input}" text)
        if(text MATCHES ";")
            message(FATAL_ERROR "${file}: schemas can't hold semicolons")
        endif()
        string(
            REPLACE "[]"
            "_array"
            text
            "${text}"
        )
        string(
            REPLACE "["
            "<"
            text
            "${text}"
        )
        string(
            REPLACE "]"
            ">"
            text
            "${text}"
        )
        string(
            REGEX REPLACE "#[^\n]*"
            ""
            text
            "${text}"
        )
        string(
            REPLACE "\n"
            ";"
            raw
            "${text}"
        )
        set(lines "")
        set(numbers "")
        set(number 0)
        set(open "")
        foreach(line IN LISTS raw)
            math(EXPR number "${number} + 1")
            string(STRIP "${line}" line)
            if(line STREQUAL "")
                continue()
            endif()
            if(NOT open STREQUAL "")
                if(line MATCHES "^\\)(.*)$")
                    list(JOIN items ", " joined)
                    list(APPEND lines "${open}(${joined})${CMAKE_MATCH_1}")
                    list(APPEND numbers ${open_number})
                    set(open "")
                else()
                    string(
                        REGEX REPLACE ",[ \t]*$"
                        ""
                        line
                        "${line}"
                    )
                    list(APPEND items "${line}")
                endif()
                continue()
            endif()
            if(line MATCHES "^(.*[^ \t])[ \t]*\\($")
                set(open "${CMAKE_MATCH_1} ")
                set(open_number ${number})
                set(items "")
                continue()
            endif()
            list(APPEND lines "${line}")
            list(APPEND numbers ${number})
        endforeach()
        if(NOT open STREQUAL "")
            message(FATAL_ERROR "${file}:${open_number}: metadata opened with ( is never closed")
        endif()
        set(expanded "")
        set(expanded_numbers "")
        foreach(line number IN ZIP_LISTS lines numbers)
            if(line MATCHES "^texturable[ \t]")
                _ernie_schema_texturable("${line}" "${file}:${number}" generated)
                list(APPEND expanded ${generated})
                foreach(unused IN LISTS generated)
                    list(APPEND expanded_numbers ${number})
                endforeach()
            else()
                list(APPEND expanded "${line}")
                list(APPEND expanded_numbers ${number})
            endif()
        endforeach()
        set(lines ${expanded})
        set(numbers ${expanded_numbers})
        set(current "")
        foreach(line number IN ZIP_LISTS lines numbers)
            set(where "${file}:${number}")
            if(current STREQUAL "")
                if(
                    NOT line
                        MATCHES
                        "^(abstract |api |multiple api )?class ([A-Za-z_][A-Za-z0-9_]*)[ \t]*(:[^({]*)?(\\((.*)\\))?[ \t]*{[ \t]*(})?$"
                )
                    message(FATAL_ERROR "${where}: expected a class")
                endif()
                set(qualifier "${CMAKE_MATCH_1}")
                set(name "${CMAKE_MATCH_2}")
                set(parents "${CMAKE_MATCH_3}")
                set(metadata "${CMAKE_MATCH_5}")
                set(closed "${CMAKE_MATCH_6}")
                if(DEFINED kind_${name})
                    message(FATAL_ERROR "${where}: ${name} is declared twice")
                endif()
                if(qualifier STREQUAL "abstract ")
                    set(kind_${name} Abstract)
                elseif(qualifier MATCHES "api $")
                    set(kind_${name} Api)
                else()
                    set(kind_${name} Concrete)
                endif()
                set(multiple_${name} FALSE)
                if(qualifier STREQUAL "multiple api ")
                    set(multiple_${name} TRUE)
                endif()
                set(base_${name} None)
                set(apis_${name} "")
                set(fields_${name} "")
                _ernie_schema_metadata(
                    "${metadata}"
                    "${where}"
                    FALSE
                    flags
                    doc
                    unit
                    allowed
                    enum
                    group
                    display
                    enabled
                    min
                    max
                    applies
                    auto
                )
                set(doc_${name} "${doc}")
                if((applies OR auto) AND NOT kind_${name} STREQUAL "Api")
                    message(FATAL_ERROR "${where}: only APIs take appliesTo and autoApplyTo")
                endif()
                set(applies_${name} "${applies}")
                set(auto_${name} "${auto}")
                string(
                    REGEX REPLACE "^:"
                    ""
                    parents
                    "${parents}"
                )
                string(
                    REPLACE ","
                    ";"
                    parents
                    "${parents}"
                )
                set(instances_${name} "")
                foreach(parent IN LISTS parents)
                    string(STRIP "${parent}" parent)
                    if(parent MATCHES "^([A-Za-z_][A-Za-z0-9_]*):([A-Za-z_][A-Za-z0-9_]*)$")
                        set(api "${CMAKE_MATCH_1}")
                        if(NOT DEFINED kind_${api} OR NOT multiple_${api})
                            message(FATAL_ERROR "${where}: ${api} isn't an API applied many times")
                        endif()
                        list(APPEND instances_${name} "${parent}")
                        continue()
                    endif()
                    if(NOT DEFINED kind_${parent})
                        message(FATAL_ERROR "${where}: ${parent} isn't declared before ${name}")
                    endif()
                    if(kind_${name} STREQUAL "Api" AND NOT kind_${parent} STREQUAL "Api")
                        message(FATAL_ERROR "${where}: API ${name} can't have a base, only APIs")
                    endif()
                    if(multiple_${parent})
                        message(
                            FATAL_ERROR
                            "${where}: ${parent} is applied many times, so name an instance as ${parent}:name"
                        )
                    endif()
                    if(kind_${parent} STREQUAL "Api")
                        list(APPEND apis_${name} ${parent})
                    elseif(NOT base_${name} STREQUAL "None")
                        message(FATAL_ERROR "${where}: ${name} has two bases")
                    else()
                        set(base_${name} ${parent})
                    endif()
                endforeach()
                list(APPEND schemas ${name})
                set(family_${name} ${family})
                if(NOT closed)
                    set(current ${name})
                endif()
                continue()
            endif()
            if(line STREQUAL "}")
                if(multiple_${current})
                    foreach(field instance next)
                        if(field IN_LIST fields_${current})
                            message(
                                FATAL_ERROR
                                "${where}: ${current} declares ${field}, which instances use"
                            )
                        endif()
                        list(APPEND fields_${current} ${field})
                        set(attribute_${current}_${field} "ernie:${field}")
                        set(flags_${current}_${field} internal)
                        set(unit_${current}_${field} None)
                        set(group_${current}_${field} "")
                        set(display_${current}_${field} "")
                        set(enabled_${current}_${field} "")
                        set(min_${current}_${field} 0)
                        set(max_${current}_${field} 0)
                        set(where_${current}_${field} "${where}")
                    endforeach()
                    set(type_${current}_instance token)
                    set(fallback_${current}_instance "{}")
                    set(doc_${current}_instance "The instance name the row belongs to.")
                    set(type_${current}_next uint)
                    set(fallback_${current}_next "0xFFFFFFFFu")
                    set(doc_${current}_next "The row of the prim's next instance.")
                endif()
                set(current "")
                continue()
            endif()
            if(NOT line MATCHES "^([a-z0-9_]+)[ \t]+([A-Za-z_][A-Za-z0-9_:]*)(.*)$")
                message(FATAL_ERROR "${where}: expected an attribute or }")
            endif()
            set(type "${CMAKE_MATCH_1}")
            set(attribute "${CMAKE_MATCH_2}")
            set(rest "${CMAKE_MATCH_3}")
            if(rest MATCHES "\\([ \t]*derived[ \t]*(,|\\))")
                message(
                    FATAL_ERROR
                    "${where}: ${attribute} is derived, but schemas hold only authored attributes. A system computes derived data into its own table."
                )
            endif()
            set(metadata "")
            if(rest MATCHES "^(.*)\\(([ \t]*(${_ernie_schema_metadata_keys}).*)\\)[ \t]*$")
                set(rest "${CMAKE_MATCH_1}")
                set(metadata "${CMAKE_MATCH_2}")
            endif()
            _ernie_schema_metadata(
                "${metadata}"
                "${where}"
                TRUE
                flags
                doc
                unit
                allowed
                enum
                group
                display
                enabled
                min
                max
                applies
                auto
            )
            string(STRIP "${rest}" rest)
            set(value "")
            if(rest MATCHES "^=[ \t]*(.+)$")
                string(STRIP "${CMAKE_MATCH_1}" value)
            elseif(NOT rest STREQUAL "")
                message(FATAL_ERROR "${where}: expected = and a fallback after ${attribute}")
            endif()
            set(overridden FALSE)
            foreach(entry IN LISTS instances_${current})
                string(
                    REPLACE ":"
                    ";"
                    parts
                    "${entry}"
                )
                list(GET parts 0 api)
                list(GET parts 1 instance)
                foreach(field IN LISTS fields_${api})
                    string(
                        REPLACE "__INSTANCE_NAME__"
                        "${instance}"
                        named
                        "${attribute_${api}_${field}}"
                    )
                    if(named STREQUAL attribute)
                        _ernie_schema_value(${type_${api}_${field}} "${value}" "${where}" fallback)
                        set(override_${current}_${instance}_${field} "${fallback}")
                        list(APPEND overrides_${current}_${instance} ${field})
                        set(overridden TRUE)
                    endif()
                endforeach()
            endforeach()
            if(overridden)
                continue()
            endif()
            if(multiple_${current})
                if(NOT attribute MATCHES "(^|:)__INSTANCE_NAME__(:|$)")
                    message(
                        FATAL_ERROR
                        "${where}: ${current} is applied many times, so ${attribute} must name __INSTANCE_NAME__"
                    )
                endif()
                string(
                    REGEX REPLACE "(^|:)__INSTANCE_NAME__(:|$)"
                    "\\1"
                    named
                    "${attribute}"
                )
                string(
                    REGEX REPLACE ":$"
                    ""
                    named
                    "${named}"
                )
                _ernie_schema_field("${named}" field)
            elseif(attribute MATCHES "__INSTANCE_NAME__")
                message(FATAL_ERROR "${where}: only APIs applied many times name instances")
            else()
                _ernie_schema_field("${attribute}" field)
            endif()
            if(field IN_LIST fields_${current})
                message(FATAL_ERROR "${where}: ${current} has two ${field}s")
            endif()
            if(allowed OR enum)
                if(NOT type STREQUAL "token")
                    message(FATAL_ERROR "${where}: only tokens take allowed values or an enum")
                endif()
                if(NOT enum)
                    _ernie_schema_capitalize("${field}" enum)
                endif()
            endif()
            if(enum AND DEFINED _ernie_schema_allowed_${enum})
                if(allowed AND NOT allowed STREQUAL _ernie_schema_allowed_${enum})
                    message(
                        FATAL_ERROR
                        "${where}: the enum ${enum} has other values; name this one with enum = Name"
                    )
                endif()
                set(type "enum_${enum}")
            elseif(enum)
                if(NOT allowed)
                    message(
                        FATAL_ERROR
                        "${where}: the enum ${enum} isn't declared; give its allowed values"
                    )
                endif()
                set(_ernie_schema_allowed_${enum} "${allowed}")
                set(names "")
                set(quoted "")
                foreach(token IN LISTS allowed)
                    _ernie_schema_enum_name("${token}" name)
                    list(APPEND names "    ${name},\n")
                    list(APPEND quoted "\"${token}\"")
                endforeach()
                list(JOIN names "" names)
                list(JOIN quoted ", " quoted)
                list(LENGTH allowed count)
                string(
                    APPEND gpu_${family}
                    "ERNIE_ENUM(${enum}) {\n${names}};\n"
                    "static const uint k${enum}Count = ${count};\n\n"
                )
                string(
                    APPEND tokens_text
                    "inline constexpr std::string_view k${enum}Tokens[] = {${quoted}};\n"
                )
                set(type "enum_${enum}")
            endif()
            _ernie_schema_types(${type} "${where}" gpu cpp)
            _ernie_schema_value(${type} "${value}" "${where}" fallback)
            list(APPEND fields_${current} ${field})
            set(type_${current}_${field} ${type})
            set(attribute_${current}_${field} "${attribute}")
            set(flags_${current}_${field} "${flags}")
            set(doc_${current}_${field} "${doc}")
            set(unit_${current}_${field} "${unit}")
            set(group_${current}_${field} "${group}")
            set(display_${current}_${field} "${display}")
            set(enabled_${current}_${field} "${enabled}")
            set(min_${current}_${field} "${min}")
            set(max_${current}_${field} "${max}")
            set(fallback_${current}_${field} "${fallback}")
            set(where_${current}_${field} "${where}")
        endforeach()
        if(NOT current STREQUAL "")
            message(FATAL_ERROR "${file}: ${current} isn't closed")
        endif()
    endforeach()

    set(api_list "")
    foreach(name IN LISTS schemas)
        if(kind_${name} STREQUAL "Api")
            list(LENGTH api_list bit_${name})
            list(APPEND api_list ${name})
        endif()
    endforeach()
    list(LENGTH api_list api_total)
    if(api_total GREATER 64)
        message(FATAL_ERROR "${api_total} APIs, but a prim's API mask holds 64")
    endif()
    foreach(api IN LISTS api_list)
        foreach(target IN LISTS applies_${api} auto_${api})
            if(NOT DEFINED kind_${target} OR kind_${target} STREQUAL "Api")
                message(FATAL_ERROR "${api} names ${target}, which isn't a typed schema")
            endif()
        endforeach()
        foreach(target IN LISTS auto_${api})
            if(NOT api IN_LIST apis_${target})
                list(APPEND apis_${target} ${api})
            endif()
        endforeach()
    endforeach()
    foreach(name IN LISTS schemas)
        set(expanded_${name} ${apis_${name}})
        set(index 0)
        list(LENGTH expanded_${name} length)
        while(index LESS length)
            list(GET expanded_${name} ${index} api)
            foreach(inner IN LISTS apis_${api})
                if(NOT inner IN_LIST expanded_${name})
                    list(APPEND expanded_${name} ${inner})
                endif()
            endforeach()
            math(EXPR index "${index} + 1")
            list(LENGTH expanded_${name} length)
        endwhile()
    endforeach()
    foreach(name IN LISTS schemas)
        foreach(api IN LISTS expanded_${name})
            if(applies_${api} AND NOT kind_${name} STREQUAL "Api")
                _ernie_schema_is_a(${name} "${applies_${api}}" allowed)
                if(NOT allowed)
                    message(
                        FATAL_ERROR
                        "${name} builds in ${api}, which only applies to ${applies_${api}}"
                    )
                endif()
            endif()
        endforeach()
    endforeach()

    set(ids "")
    set(bases "")
    set(info "")
    set(apis "")
    set(api_count 0)
    set(columns_text "")
    set(column_refs_text "")
    set(column_count 0)
    set(slots_text "")
    set(slot_count 0)
    set(slot_constants "")
    set(schemas_text "")
    set(schemas_bytes 0)
    set(fallbacks_text "")
    set(python_prims "")
    set(python_selections "")
    set(python_names "")
    set(define_declarations "")
    set(typed_text "")
    set(define_definitions "")
    set(python_define_cases "")
    set(python_apply_cases "")
    set(python_assign_functions "")
    set(api_ids_text "")
    set(api_schemas_text "")
    set(api_targets_text "")
    set(api_target_count 0)
    set(slot_counts_text "")
    set(builtin_instances_text "")
    set(builtin_instance_count 0)
    set(multiple_mask "")
    foreach(api IN LISTS api_list)
        string(APPEND api_ids_text "static const uint kApi${api} = ${bit_${api}};\n")
        string(APPEND api_schemas_text "    SchemaId::${api},\n")
        if(multiple_${api})
            list(APPEND multiple_mask "(uint64_t(1) << ${bit_${api}})")
        endif()
    endforeach()
    list(JOIN multiple_mask " | " multiple_mask)
    if(NOT multiple_mask)
        set(multiple_mask 0)
    endif()
    foreach(name IN LISTS schemas)
        set(family ${family_${name}})
        string(APPEND ids "    ${name},\n")
        string(APPEND bases "    uint(SchemaId::${base_${name}}),\n")

        set(chain ${name})
        set(level ${name})
        while(NOT base_${level} STREQUAL "None")
            set(level ${base_${level}})
            list(PREPEND chain ${level})
        endwhile()
        set(desc "")
        set(columns "")
        set(typed "")
        set(builtin "")
        foreach(level IN LISTS chain)
            foreach(api IN LISTS expanded_${level})
                if(NOT api IN_LIST builtin)
                    list(APPEND builtin ${api})
                endif()
            endforeach()
        endforeach()
        set(members "")
        foreach(level IN LISTS chain builtin)
            foreach(entry IN LISTS instances_${level})
                list(APPEND members "${level}:${entry}")
                string(
                    REGEX REPLACE ":.*$"
                    ""
                    api
                    "${entry}"
                )
                if(NOT api IN_LIST builtin)
                    list(APPEND builtin ${api})
                endif()
            endforeach()
        endforeach()
        set(optional "")
        if(NOT kind_${name} STREQUAL "Api")
            foreach(api IN LISTS api_list)
                if(applies_${api} AND NOT api IN_LIST builtin)
                    _ernie_schema_is_a(${name} "${applies_${api}}" allowed)
                    if(allowed)
                        list(FIND schemas ${api} api_index)
                        list(FIND schemas ${name} name_index)
                        if(api_index GREATER name_index)
                            message(
                                FATAL_ERROR
                                "${api} applies to ${name}, so declare it before ${name}"
                            )
                        endif()
                        list(APPEND optional ${api})
                    endif()
                endif()
            endforeach()
        endif()
        set(mask "")
        foreach(api IN LISTS builtin)
            list(APPEND mask "(uint64_t(1) << ${bit_${api}})")
        endforeach()
        list(JOIN mask " | " mask)
        if(NOT mask)
            set(mask 0)
        endif()
        set(target_first ${api_target_count})
        set(api_bit "kNoApi")
        if(kind_${name} STREQUAL "Api")
            set(api_bit ${bit_${name}})
            foreach(target IN LISTS applies_${name})
                string(APPEND api_targets_text "    SchemaId::${target},\n")
                math(EXPR api_target_count "${api_target_count} + 1")
            endforeach()
        endif()
        math(EXPR target_count "${api_target_count} - ${target_first}")
        foreach(level IN LISTS chain)
            foreach(owner IN LISTS expanded_${level} level)
                foreach(field IN LISTS fields_${owner})
                    if(field IN_LIST typed OR type_${owner}_${field} STREQUAL "inputs")
                        continue()
                    endif()
                    list(APPEND typed ${field})
                    set(typed_owner_${field} ${owner})
                    set(typed_type_${field} ${type_${owner}_${field}})
                endforeach()
            endforeach()
            foreach(api IN LISTS expanded_${level})
                foreach(field IN LISTS fields_${api})
                    if("internal" IN_LIST flags_${api}_${field})
                        continue()
                    endif()
                    if(field IN_LIST desc)
                        message(
                            FATAL_ERROR
                            "${where_${api}_${field}}: ${api}'s ${field} collides in ${name}"
                        )
                    endif()
                    list(APPEND desc ${field})
                    set(desc_type_${field} ${type_${api}_${field}})
                    set(desc_owner_${field} ${api})
                    set(desc_fallback_${field} "${fallback_${api}_${field}}")
                endforeach()
            endforeach()
            foreach(field IN LISTS fields_${level})
                if("internal" IN_LIST flags_${level}_${field})
                    if(level STREQUAL name)
                        list(APPEND columns ${field})
                    endif()
                    continue()
                endif()
                if(field IN_LIST desc)
                    if(NOT type_${level}_${field} STREQUAL desc_type_${field})
                        message(
                            FATAL_ERROR
                            "${where_${level}_${field}}: ${field} overrides a ${desc_type_${field}}"
                        )
                    endif()
                else()
                    list(APPEND desc ${field})
                    set(desc_type_${field} ${type_${level}_${field}})
                    set(desc_owner_${field} ${level})
                    if(level STREQUAL name AND NOT type_${level}_${field} STREQUAL "inputs")
                        list(APPEND columns ${field})
                    endif()
                endif()
                set(desc_fallback_${field} "${fallback_${level}_${field}}")
            endforeach()
        endforeach()

        set(count 1)
        set(first_column ${column_count})
        set(stamps_gpu "")
        string(APPEND gpu_${family} "struct ${name}Data {\n")
        foreach(field IN LISTS columns)
            set(enabled ${enabled_${name}_${field}})
            if(enabled AND NOT enabled IN_LIST desc)
                message(
                    FATAL_ERROR
                    "${where_${name}_${field}}: ${field} is enabled by ${enabled}, which ${name} doesn't have"
                )
            endif()
            _ernie_schema_types(${type_${name}_${field}} "" gpu cpp)
            _ernie_schema_column(${type_${name}_${field}} column role)
            set(tokens "{}")
            if(type_${name}_${field} MATCHES "^enum_(.+)$")
                set(tokens "k${CMAKE_MATCH_1}Tokens")
            endif()
            string(APPEND gpu_${family} "    Paged<${gpu}> ${field};\n")
            math(EXPR index_${name}_${field} "${column_count} - ${first_column}")
            _ernie_schema_capitalize("${field}" constant)
            string(
                APPEND column_refs_text
                "inline constexpr ColumnRef k${name}${constant} = {SchemaId::${name}, &kColumns[${column_count}], ${index_${name}_${field}}};\n"
            )
            set(flags "")
            foreach(flag IN LISTS flags_${name}_${field})
                string(SUBSTRING "${flag}" 0 1 first)
                string(SUBSTRING "${flag}" 1 -1 others)
                string(TOUPPER "${first}" first)
                list(APPEND flags "ColumnFlags::${first}${others}")
            endforeach()
            if(NOT flags)
                set(flags "ColumnFlags::None")
            endif()
            list(JOIN flags " | " flags)
            set(stamp 0)
            if("tracked" IN_LIST flags_${name}_${field})
                string(APPEND stamps_gpu "    Paged<uint64_t> ${field}Stamp;\n")
                set(stamp "offsetof(${name}Data, ${field}Stamp)")
                math(EXPR count "${count} + 1")
            endif()
            string(
                APPEND columns_text
                "    {\"${field}\", \"${attribute_${name}_${field}}\", ColumnType::${column}, ColumnRole::${role},\n"
                "     ${flags}, offsetof(${name}Data, ${field}), ${stamp}, SchemaId::None,\n"
                "     \"${doc_${name}_${field}}\", Unit::${unit_${name}_${field}}, ${tokens},\n"
                "     \"${group_${name}_${field}}\", \"${display_${name}_${field}}\",\n"
                "     float(${min_${name}_${field}}), float(${max_${name}_${field}}),\n"
                "     \"${enabled_${name}_${field}}\"},\n"
            )
            math(EXPR count "${count} + 1")
            math(EXPR column_count "${column_count} + 1")
        endforeach()
        math(EXPR own_columns "${column_count} - ${first_column}")
        set(columns_${name} ${own_columns})

        set(slots "")
        foreach(level IN LISTS chain)
            if(columns_${level} GREATER 0 AND NOT kind_${level} STREQUAL "Api")
                list(APPEND slots ${level})
            endif()
        endforeach()
        set(first_slot ${slot_count})
        set(slot 0)
        foreach(entry IN LISTS slots)
            if(entry STREQUAL name)
                string(APPEND slot_constants "static const uint kSlot${name} = ${slot};\n")
            endif()
            string(APPEND slots_text "    SchemaId::${entry},\n")
            math(EXPR slot "${slot} + 1")
            math(EXPR slot_count "${slot_count} + 1")
        endforeach()
        list(LENGTH slots own_slots)
        string(APPEND slot_counts_text "    ${own_slots},\n")
        math(EXPR bytes "${count} * 8")
        string(
            APPEND gpu_${family}
            "${stamps_gpu}    Paged<uint> prim;\n};\nERNIE_CHECK_SIZE(${name}Data, ${bytes});\n\n"
        )
        string(SUBSTRING "${name}" 0 1 first)
        string(SUBSTRING "${name}" 1 -1 others)
        string(TOLOWER "${first}" first)
        string(APPEND schemas_text "    ${name}Data ${first}${others};\n")
        math(EXPR schemas_bytes "${schemas_bytes} + ${bytes}")

        list(LENGTH apis_${name} own)
        foreach(api IN LISTS apis_${name})
            string(APPEND apis "    SchemaId::${api},\n")
        endforeach()
        string(
            APPEND info
            "    {\"${name}\", SchemaId::${base_${name}}, SchemaKind::${kind_${name}}, ${api_count}, ${own},\n"
            "     ${first_column}, ${own_columns}, sizeof(${name}Data), ${first_slot}, ${own_slots},\n"
            "     ${api_bit}, ${mask}, ${target_first}, ${target_count}, \"${doc_${name}}\"},\n"
        )
        math(EXPR api_count "${api_count} + ${own}")

        set(constants "")
        set(bit_constant "")
        if(kind_${name} STREQUAL "Api")
            string(
                CONCAT bit_constant
                "    static constexpr uint64_t kBit = uint64_t(1) << ${bit_${name}};\n"
                "    static constexpr uint64_t kApis = kBit | ${mask};\n"
            )
        endif()
        set(prim_accessors "")
        set(view_accessors "")
        foreach(field IN LISTS typed)
            set(owner ${typed_owner_${field}})
            set(type ${typed_type_${field}})
            _ernie_schema_types(${type} "" gpu cpp)
            _ernie_schema_capitalize("${field}" capital)
            string(
                APPEND constants
                "    static constexpr Field<${cpp}> k${capital} = {SchemaId::${owner}, ${index_${owner}_${field}}};\n"
            )
            if(type MATCHES "^(token|asset)$")
                set(attribute "Attribute<Token>")
            elseif(type MATCHES "^(prim_array|token_array|primvars)$")
                set(attribute "UntypedAttribute")
            elseif(type MATCHES "_array$")
                string(
                    REGEX REPLACE "^std::vector<(.+)>$"
                    "\\1"
                    item
                    "${cpp}"
                )
                set(attribute "ArrayAttribute<${item}>")
            else()
                set(attribute "Attribute<${cpp}>")
            endif()
            string(
                REGEX REPLACE "(^|<)([A-Z][A-Za-z0-9]*)"
                "\\1ernie::\\2"
                cpp
                "${cpp}"
            )
            string(
                REGEX REPLACE "(^|<)([A-Z][A-Za-z0-9]*)"
                "\\1ernie::\\2"
                attribute
                "${attribute}"
            )
            string(
                APPEND prim_accessors
                "\n    ${cpp} ${capital}() const {\n        return Get(${name}::k${capital});\n    }\n"
            )
            if(NOT "internal" IN_LIST flags_${owner}_${field})
                string(
                    APPEND prim_accessors
                    "\n    void Set${capital}(const ${cpp}& value) const {\n"
                    "        Set(${name}::k${capital}, value);\n    }\n"
                )
            endif()
            string(
                APPEND view_accessors
                "\n    ${attribute} ${capital}() const {\n        return Get(${name}::k${capital});\n    }\n"
            )
        endforeach()
        string(
            APPEND typed_text
            "class ${name}Prim : public Prim {\n  public:\n    ${name}Prim() = default;\n\n"
            "    explicit ${name}Prim(const Prim& prim) : Prim(prim) {\n"
            "        CheckSchema(${name}::kSchema);\n    }\n${prim_accessors}};\n\n"
            "class ${name}View : public View {\n  public:\n    ${name}View() = default;\n\n"
            "    explicit ${name}View(View view) : View(std::move(view)) {}\n\n"
            "    ${name}View World(uint32_t world) const {\n"
            "        return ${name}View(View::World(world));\n    }\n${view_accessors}};\n\n"
        )
        string(
            APPEND desc_${family}
            "class ${name}Prim;\nclass ${name}View;\n\n"
            "struct ${name} {\n    static constexpr SchemaId kSchema = SchemaId::${name};\n${bit_constant}"
            "    using PrimType = ${name}Prim;\n    using ViewType = ${name}View;\n\n"
            "${constants}\n    struct Desc {\n"
        )
        foreach(field IN LISTS desc)
            _ernie_schema_types(${desc_type_${field}} "" gpu cpp)
            string(APPEND desc_${family} "        ${cpp} ${field} = ${desc_fallback_${field}};\n")
        endforeach()
        set(applied "")
        foreach(api IN LISTS optional)
            string(
                REGEX REPLACE "API$"
                ""
                member
                "${api}"
            )
            string(SUBSTRING "${member}" 0 1 first)
            string(SUBSTRING "${member}" 1 -1 others)
            string(TOLOWER "${first}" first)
            set(member "${first}${others}")
            if(member IN_LIST desc)
                message(FATAL_ERROR "${name} has a ${member} and applies ${api} under that name")
            endif()
            set(member_${name}_${api} ${member})
            if(multiple_${api})
                set(stored "")
                foreach(field IN LISTS fields_${api})
                    if(NOT "internal" IN_LIST flags_${api}_${field})
                        list(APPEND stored ${field})
                    endif()
                endforeach()
                list(LENGTH stored stored_count)
                if(stored_count EQUAL 1)
                    _ernie_schema_types(${type_${api}_${stored}} "" gpu cpp)
                    set(single_${name}_${api} ${stored})
                else()
                    set(cpp "${api}::Desc")
                    set(single_${name}_${api} "")
                endif()
                string(
                    APPEND desc_${family}
                    "        std::map<std::string, ${cpp}, std::less<>> ${member};\n"
                )
            else()
                string(APPEND desc_${family} "        std::optional<${api}::Desc> ${member};\n")
                list(APPEND applied "(d.${member} ? ${api}::kApis : 0)")
            endif()
        endforeach()
        foreach(entry IN LISTS members)
            string(
                REPLACE ":"
                ";"
                parts
                "${entry}"
            )
            list(GET parts 0 level)
            list(GET parts 1 api)
            list(GET parts 2 instance)
            if(NOT kind_${name} STREQUAL "Api")
                string(
                    APPEND builtin_instances_text
                    "    {SchemaId::${name}, SchemaId::${api}, \"${instance}\"},\n"
                )
                math(EXPR builtin_instance_count "${builtin_instance_count} + 1")
            endif()
            if(instance IN_LIST desc OR instance IN_LIST optional)
                message(FATAL_ERROR "${name} has a ${instance} and builds in ${api}:${instance}")
            endif()
            set(initial "")
            foreach(field IN LISTS fields_${api})
                if(field IN_LIST overrides_${level}_${instance})
                    string(
                        APPEND initial
                        " d.${field} = ${override_${level}_${instance}_${field}};"
                    )
                endif()
            endforeach()
            if(initial STREQUAL "")
                string(APPEND desc_${family} "        ${api}::Desc ${instance};\n")
            else()
                string(
                    APPEND desc_${family}
                    "        ${api}::Desc ${instance} = [] { ${api}::Desc d;${initial} return d; }();\n"
                )
            endif()
        endforeach()
        list(JOIN applied " | " applied)
        string(APPEND desc_${family} "    };\n\n")
        if(applied)
            string(
                APPEND desc_${family}
                "    static uint64_t Apis(const Desc& d) {\n        return ${applied};\n    }\n\n"
            )
        else()
            string(
                APPEND desc_${family}
                "    static uint64_t Apis(const Desc&) {\n        return 0;\n    }\n\n"
            )
        endif()
        if(desc OR optional OR members)
            string(
                APPEND desc_${family}
                "    template <class W> static void Write(W& w, const Desc& d) {\n"
            )
            foreach(field IN LISTS desc)
                if(desc_type_${field} STREQUAL "inputs")
                    continue()
                endif()
                set(owner ${desc_owner_${field}})
                string(
                    APPEND desc_${family}
                    "        w.Set(SchemaId::${owner}, ${index_${owner}_${field}}, d.${field});\n"
                )
            endforeach()
            foreach(api IN LISTS optional)
                set(member ${member_${name}_${api}})
                if(NOT multiple_${api})
                    string(
                        APPEND desc_${family}
                        "        if (d.${member}) {\n"
                        "            ${api}::Write(w, *d.${member});\n        }\n"
                    )
                elseif(single_${name}_${api})
                    string(
                        APPEND desc_${family}
                        "        for (const auto& [instance, value] : d.${member}) {\n"
                        "            w.Instance(SchemaId::${api}, instance, [&](W& i) {\n"
                        "                i.Set(SchemaId::${api}, ${index_${api}_${single_${name}_${api}}}, value);\n"
                        "            });\n        }\n"
                    )
                else()
                    string(
                        APPEND desc_${family}
                        "        for (const auto& [instance, value] : d.${member}) {\n"
                        "            w.Instance(SchemaId::${api}, instance, [&](W& i) {\n"
                        "                ${api}::Write(i, value);\n"
                        "            });\n        }\n"
                    )
                endif()
            endforeach()
            foreach(entry IN LISTS members)
                string(
                    REPLACE ":"
                    ";"
                    parts
                    "${entry}"
                )
                list(GET parts 1 api)
                list(GET parts 2 instance)
                string(
                    APPEND desc_${family}
                    "        w.Instance(SchemaId::${api}, \"${instance}\", [&](W& i) {\n"
                    "            ${api}::Write(i, d.${instance});\n        });\n"
                )
            endforeach()
            string(APPEND desc_${family} "    }\n};\n\n")
        else()
            string(
                APPEND desc_${family}
                "    template <class W> static void Write(W&, const Desc&) {}\n};\n\n"
            )
        endif()
        string(
            APPEND fallbacks_text
            "    case SchemaId::${name}:\n        ${name}::Write(w, {});\n        return;\n"
        )
        if(kind_${name} STREQUAL "Concrete")
            string(
                APPEND define_declarations
                "ERNIE_API ${name}Prim Define(Scene& scene, std::string_view path, const ${name}::Desc& desc = {});\n"
                "ERNIE_API PrimRange Define(Scene& scene, Span<const std::string> paths,\n"
                "                           const ${name}::Desc& desc = {}, const BatchDesc& batch = {});\n"
            )
            set(inputs "")
            if("inputs" IN_LIST desc AND desc_type_inputs STREQUAL "inputs")
                set(inputs "\n        SetInputs(writer, ${name}::kSchema, desc.inputs);")
            endif()
            set(assignments "")
            foreach(field IN LISTS desc)
                if(desc_type_${field} STREQUAL "inputs")
                    continue()
                endif()
                set(tokens "")
                if(desc_type_${field} MATCHES "^enum_(.+)$")
                    set(tokens ", k${CMAKE_MATCH_1}Tokens")
                endif()
                string(
                    APPEND assignments
                    "            if (field == \"${field}\") {\n"
                    "                Assign(desc.${field}, value${tokens});\n"
                    "                continue;\n"
                    "            }\n"
                )
            endforeach()
            foreach(api IN LISTS optional)
                if(NOT multiple_${api})
                    continue()
                endif()
                set(member ${member_${name}_${api}})
                if(single_${name}_${api})
                    set(tokens "")
                    if(type_${api}_${single_${name}_${api}} MATCHES "^enum_(.+)$")
                        set(tokens ", k${CMAKE_MATCH_1}Tokens")
                    endif()
                    string(
                        APPEND assignments
                        "            if (field == \"${member}\") {\n"
                        "                for (auto [instance, entry] : nanobind::cast<nanobind::dict>(value)) {\n"
                        "                    Assign(desc.${member}[nanobind::cast<std::string>(instance)], entry${tokens});\n"
                        "                }\n"
                        "                continue;\n"
                        "            }\n"
                    )
                else()
                    string(
                        APPEND assignments
                        "            if (field == \"${member}\") {\n"
                        "                for (auto [instance, entry] : nanobind::cast<nanobind::dict>(value)) {\n"
                        "                    AssignDesc(desc.${member}[nanobind::cast<std::string>(instance)],\n"
                        "                               nanobind::cast<nanobind::dict>(entry));\n"
                        "                }\n"
                        "                continue;\n"
                        "            }\n"
                    )
                endif()
            endforeach()
            foreach(entry IN LISTS members)
                string(
                    REPLACE ":"
                    ";"
                    parts
                    "${entry}"
                )
                list(GET parts 2 instance)
                string(
                    APPEND assignments
                    "            if (field == \"${instance}\") {\n"
                    "                AssignDesc(desc.${instance}, nanobind::cast<nanobind::dict>(value));\n"
                    "                continue;\n"
                    "            }\n"
                )
            endforeach()
            set(api_assignments "")
            foreach(api IN LISTS optional)
                if(multiple_${api})
                    continue()
                endif()
                string(
                    APPEND api_assignments
                    "            if (api == \"${api}\") {\n"
                    "                AssignDesc(desc.${member_${name}_${api}}.emplace(), nanobind::cast<nanobind::dict>(value));\n"
                    "                continue;\n"
                    "            }\n"
                )
            endforeach()
            string(
                APPEND python_define_cases
                "    case SchemaId::${name}: {\n"
                "        ${name}::Desc desc;\n"
                "        for (auto [key, value] : values) {\n"
                "            const std::string field = nanobind::cast<std::string>(key);\n"
                "${assignments}"
                "            throw nanobind::key_error((\"${name} has no \" + field).c_str());\n"
                "        }\n"
                "        for (auto [key, value] : apis) {\n"
                "            const std::string api = nanobind::cast<std::string>(key);\n"
                "${api_assignments}"
                "            throw nanobind::key_error((\"${name} isn't defined with \" + api).c_str());\n"
                "        }\n"
                "        return Define(scene, path, desc);\n"
                "    }\n"
            )
            string(
                APPEND define_definitions
                "\n${name}Prim Define(Scene& scene, std::string_view path, const ${name}::Desc& desc) {\n"
                "    return ${name}Prim(SceneImpl::Get(scene).Define(path, ${name}::kSchema, ${name}::Apis(desc),\n"
                "                                                    [&](RowWriter& writer) {\n"
                "        ${name}::Write(writer, desc);${inputs}\n"
                "    }));\n}\n\n"
                "PrimRange Define(Scene& scene, Span<const std::string> paths, const ${name}::Desc& desc,\n"
                "                 const BatchDesc& batch) {\n"
                "    return SceneImpl::Get(scene).Define(paths, ${name}::kSchema, ${name}::Apis(desc),\n"
                "                                        [&](RowWriter& writer) {\n"
                "        ${name}::Write(writer, desc);${inputs}\n"
                "    }, batch);\n}\n"
            )
        endif()
        if(kind_${name} STREQUAL "Api")
            set(api_fields "")
            foreach(field IN LISTS desc)
                set(tokens "")
                if(desc_type_${field} MATCHES "^enum_(.+)$")
                    set(tokens ", k${CMAKE_MATCH_1}Tokens")
                endif()
                string(
                    APPEND api_fields
                    "        if (field == \"${field}\") {\n"
                    "            Assign(desc.${field}, value${tokens});\n"
                    "            continue;\n"
                    "        }\n"
                )
            endforeach()
            foreach(entry IN LISTS members)
                string(
                    REPLACE ":"
                    ";"
                    parts
                    "${entry}"
                )
                list(GET parts 2 instance)
                string(
                    APPEND api_fields
                    "        if (field == \"${instance}\") {\n"
                    "            AssignDesc(desc.${instance}, nanobind::cast<nanobind::dict>(value));\n"
                    "            continue;\n"
                    "        }\n"
                )
            endforeach()
            string(
                APPEND python_assign_functions
                "void AssignDesc(${name}::Desc& desc, const nanobind::dict& values) {\n"
                "    for (auto [key, value] : values) {\n"
                "        const std::string field = nanobind::cast<std::string>(key);\n"
                "${api_fields}"
                "        throw nanobind::key_error((\"${name} has no \" + field).c_str());\n"
                "    }\n}\n\n"
            )
            if(multiple_${name})
                string(
                    APPEND python_apply_cases
                    "    case SchemaId::${name}: {\n"
                    "        ${name}::Desc desc;\n"
                    "        AssignDesc(desc, values);\n"
                    "        Apply(scene, prims, instance, desc);\n"
                    "        return;\n"
                    "    }\n"
                )
                string(
                    APPEND define_declarations
                    "ERNIE_API void Apply(Scene& scene, Span<const Prim> prims, std::string_view instance,\n"
                    "                     const ${name}::Desc& desc = {});\n"
                )
                string(
                    APPEND define_definitions
                    "\nvoid Apply(Scene& scene, Span<const Prim> prims, std::string_view instance,\n"
                    "           const ${name}::Desc& desc) {\n"
                    "    SceneImpl::Get(scene).Apply(prims, ${name}::kSchema, instance, [&](RowWriter& writer) {\n"
                    "        ${name}::Write(writer, desc);\n    });\n}\n"
                )
            else()
                string(
                    APPEND python_apply_cases
                    "    case SchemaId::${name}: {\n"
                    "        ${name}::Desc desc;\n"
                    "        AssignDesc(desc, values);\n"
                    "        Apply(scene, prims, desc);\n"
                    "        return;\n"
                    "    }\n"
                )
                string(
                    APPEND define_declarations
                    "ERNIE_API void Apply(Scene& scene, Span<const Prim> prims, const ${name}::Desc& desc = {});\n"
                )
                string(
                    APPEND define_definitions
                    "\nvoid Apply(Scene& scene, Span<const Prim> prims, const ${name}::Desc& desc) {\n"
                    "    SceneImpl::Get(scene).Apply(prims, ${name}::kSchema, [&](RowWriter& writer) {\n"
                    "        ${name}::Write(writer, desc);\n    });\n}\n"
                )
            endif()
        endif()
        if(ARG_PYTHON)
            _ernie_schema_python(${name} prim selection)
            string(APPEND python_prims "\n\n${prim}")
            string(APPEND python_selections "\n\n${selection}")
            string(APPEND python_names "    \"${name}\",\n    \"${name}Selection\",\n")
        endif()
    endforeach()

    set(includes "")
    foreach(family IN LISTS families)
        string(APPEND includes "#include \"ernie/scene/kernels/schema/${family}.h\"\n")
        string(
            CONCAT text
            "// SPDX-License-Identifier: BSD-3-Clause\n"
            "// Generated from ${source_${family}} by cmake/ErnieSchema.cmake; do not edit.\n"
            "#pragma once\n\n"
            "#include \"ernie/scene/kernels/schema/types.h\"\n\n"
            "#ifndef __SLANG__\n#include <cstdint>\n#include <limits>\n#include <map>\n#include <optional>\n#include <string>\n#include <vector>\n\n"
            "#include \"ernie/scene/primvar.h\"\n#endif\n\n"
            "namespace ernie {\n\n${gpu_${family}}}\n\n"
            "#ifndef __SLANG__\n\nnamespace ernie {\n\n${desc_${family}}}\n\n#endif\n"
        )
        _ernie_schema_write("${directory}/${family}.h" text)
    endforeach()

    list(TRANSFORM families APPEND ".schema" OUTPUT_VARIABLE sources)
    list(JOIN sources ", " sources)
    string(
        CONCAT text
        "// SPDX-License-Identifier: BSD-3-Clause\n"
        "// Generated from ${sources} by cmake/ErnieSchema.cmake; do not edit.\n"
        "#pragma once\n\n"
        "#include \"ernie/scene/kernels/schema/types.h\"\n\n"
        "namespace ernie {\n\n"
        "ERNIE_ENUM(SchemaId) {\n${ids}    Count,\n    None = 0xFFFFFFFF,\n};\n\n}\n\n"
        "${includes}\n"
        "namespace ernie {\n\n"
        "struct Schemas {\n${schemas_text}};\nERNIE_CHECK_SIZE(Schemas, ${schemas_bytes});\n\n"
        "${slot_constants}\n"
        "${api_ids_text}\n"
        "static const uint kSchemaBases[] = {\n${bases}};\n\n"
        "static const uint kSchemaSlotCounts[] = {\n${slot_counts_text}};\n\n"
        "ERNIE_SHARED bool SchemaIsA(uint schema, SchemaId base) {\n"
        "    while (schema != uint(SchemaId::None)) {\n"
        "        if (schema == uint(base)) {\n"
        "            return true;\n"
        "        }\n"
        "        schema = kSchemaBases[schema];\n"
        "    }\n"
        "    return false;\n"
        "}\n\n"
        "}\n\n"
        "#ifndef __SLANG__\n\n#include <cstddef>\n\nnamespace ernie {\n\n"
        "inline constexpr std::array<SchemaId, ${api_count}> kSchemaApis = {\n${apis}};\n\n"
        "inline constexpr std::array<SchemaId, ${slot_count}> kSchemaSlots = {\n${slots_text}};\n\n"
        "inline constexpr std::array<SchemaId, ${api_total}> kApiSchemas = {\n${api_schemas_text}};\n\n"
        "inline constexpr uint64_t kMultipleApis = ${multiple_mask};\n\n"
        "inline constexpr std::array<BuiltinInstanceInfo, ${builtin_instance_count}> kBuiltinInstances = {{\n"
        "${builtin_instances_text}}};\n\n"
        "inline constexpr std::array<SchemaId, ${api_target_count}> kApiTargets = {\n${api_targets_text}};\n\n"
        "${tokens_text}\n"
        "inline constexpr std::array<ColumnInfo, ${column_count}> kColumns = {{\n${columns_text}}};\n\n"
        "inline constexpr SchemaInfo kSchemaInfo[uint32_t(SchemaId::Count)] = {\n${info}};\n\n"
        "template <class W> void WriteFallbacks(SchemaId schema, W& w) {\n"
        "    switch (schema) {\n${fallbacks_text}"
        "    default:\n        return;\n    }\n}\n\n"
        "}\n\n#endif\n"
    )
    _ernie_schema_write("${directory}/registry.h" text)

    string(
        CONCAT text
        "// SPDX-License-Identifier: BSD-3-Clause\n"
        "// Generated from ${sources} by cmake/ErnieSchema.cmake; do not edit.\n"
        "#pragma once\n\n"
        "#include \"ernie/scene/kernels/schema/schema.h\"\n\n"
        "#ifndef __SLANG__\n\nnamespace ernie {\n\n${column_refs_text}\n}\n\n#endif\n"
    )
    _ernie_schema_write("${directory}/columns.h" text)

    if(ARG_PYTHON)
        string(
            CONCAT text
            "# SPDX-License-Identifier: BSD-3-Clause\n"
            "from __future__ import annotations\n\n"
            "from collections.abc import Mapping, Sequence\n"
            "from typing import TYPE_CHECKING, Any, ClassVar, Literal\n\n"
            "from ernie._ernie import Primvar, Scene\n"
            "from ernie._typed import (\n    Prims,\n    SchemaPrim,\n    SchemaSelection,\n    Targets,\n"
            "    InstancePrim,\n    _apis,\n    _apply,\n    _fields,\n    _given,\n    _instances,\n    _targets,\n    _unapply,\n)\n"
            "\nif TYPE_CHECKING:\n"
            "    import numpy as np\n"
            "    from numpy.typing import ArrayLike, NDArray\n"
            "${python_selections}${python_prims}\n\n__all__ = [\n${python_names}]\n"
        )
        _ernie_schema_write("${ARG_PYTHON}" text)
    endif()

    if(ARG_DEFINE_HEADER)
        string(
            CONCAT text
            "// SPDX-License-Identifier: BSD-3-Clause\n"
            "// Generated from ${sources} by cmake/ErnieSchema.cmake; do not edit.\n"
            "#pragma once\n\n"
            "#include <string>\n#include <string_view>\n\n"
            "#include \"ernie/export.h\"\n"
            "#include \"ernie/scene/kernels/schema/schema.h\"\n"
            "#include \"ernie/scene/typed.h\"\n"
            "#include \"ernie/scene/types.h\"\n"
            "#include \"ernie/util/span.h\"\n\n"
            "namespace ernie {\n\nclass Prim;\nclass Scene;\n\n${define_declarations}\n}\n"
        )
        _ernie_schema_write("${ARG_DEFINE_HEADER}" text)
        string(
            CONCAT text
            "// SPDX-License-Identifier: BSD-3-Clause\n"
            "// Generated from ${sources} by cmake/ErnieSchema.cmake; do not edit.\n"
            "#include \"ernie/scene/defines.h\"\n\n"
            "#include \"ernie/scene/structure/structure.h\"\n"
            "#include \"ernie/scene/scene.h\"\n"
            "#include \"ernie/scene/scene_impl.h\"\n\n"
            "namespace ernie {\n${define_definitions}\n}\n"
        )
        _ernie_schema_write("${ARG_DEFINE_SOURCE}" text)
    endif()

    if(ARG_TYPED_HEADER)
        string(
            CONCAT text
            "// SPDX-License-Identifier: BSD-3-Clause\n"
            "// Generated from ${sources} by cmake/ErnieSchema.cmake; do not edit.\n"
            "#pragma once\n\n"
            "#include <cstdint>\n#include <string>\n#include <utility>\n#include <vector>\n\n"
            "#include \"ernie/scene/kernels/schema/schema.h\"\n"
            "#include \"ernie/scene/views/attribute.h\"\n"
            "#include \"ernie/scene/views/view.h\"\n"
            "#include \"ernie/scene/prim.h\"\n\n"
            "namespace ernie {\n\n${typed_text}}\n"
        )
        _ernie_schema_write("${ARG_TYPED_HEADER}" text)
    endif()

    if(ARG_PYTHON_DEFINES)
        string(
            CONCAT text
            "// SPDX-License-Identifier: BSD-3-Clause\n"
            "// Generated from ${sources} by cmake/ErnieSchema.cmake; do not edit.\n"
            "#include \"values.h\"\n\n"
            "namespace ernie::python {\n\n"
            "namespace {\n\n${python_assign_functions}}\n\n"
            "Prim DefineSchema(Scene& scene, std::string_view path, SchemaId schema,\n"
            "                  const nanobind::dict& values, const nanobind::dict& apis) {\n"
            "    switch (schema) {\n${python_define_cases}"
            "    default:\n"
            "        throw nanobind::value_error(\n"
            "            (std::string(Info(schema).name) + \" isn't a concrete schema\").c_str());\n"
            "    }\n}\n\n"
            "void ApplySchema(Scene& scene, Span<const Prim> prims, SchemaId schema,\n"
            "                 std::string_view instance, const nanobind::dict& values) {\n"
            "    switch (schema) {\n${python_apply_cases}"
            "    default:\n"
            "        throw nanobind::value_error(\n"
            "            (std::string(Info(schema).name) + \" isn't an API schema\").c_str());\n"
            "    }\n}\n\n}\n"
        )
        _ernie_schema_write("${ARG_PYTHON_DEFINES}" text)
    endif()
endfunction()
