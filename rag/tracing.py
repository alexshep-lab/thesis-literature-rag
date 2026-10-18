"""Трейсинг в локальный Arize Phoenix (OpenTelemetry + OpenInference). По умолчанию выключен.

    phoenix serve                                    # UI и приёмник трейсов на http://127.0.0.1:6006
    RAG_TRACE=1 python -m rag ask "вопрос" --pdf-only

Спаны узлов LangGraph даёт инструментирование OpenInference для LangChain. Ollama вызывается через
requests, а не через LangChain, поэтому спаны LLM ставятся вручную в llm.chat (llm_span).
Трейсы содержат текст фрагментов корпуса, поэтому принимаются только локальные адреса.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from urllib.parse import urlparse

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_tracer = None


def enabled() -> bool:
    return os.getenv("RAG_TRACE", "0") == "1"


def endpoint() -> str:
    return os.getenv("RAG_TRACE_ENDPOINT", "http://127.0.0.1:6006/v1/traces")


def setup() -> bool:
    """Подключить экспорт трейсов в локальный Phoenix. Без RAG_TRACE=1 ничего не делает и ничего не импортирует."""
    global _tracer
    if not enabled():
        return False
    if _tracer is not None:
        return True
    url = endpoint()
    if urlparse(url).hostname not in _LOCAL_HOSTS:
        raise RuntimeError(f"RAG_TRACE_ENDPOINT={url}: разрешён только локальный Phoenix (корпус не уходит наружу)")
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    # Свой TracerProvider вместо phoenix.otel.register: тот читает PHOENIX_COLLECTOR_ENDPOINT/PHOENIX_API_KEY
    # из окружения и может уйти в облако; здесь адрес только явный и проверенный.
    provider = TracerProvider(resource=Resource.create({
        "service.name": "thesis-literature-rag",
        "openinference.project.name": os.getenv("RAG_TRACE_PROJECT", "thesis-rag"),
    }))
    import requests

    # trust_env=False: экспорт не идёт через HTTP(S)_PROXY / системный прокси, даже если 127.0.0.1 не в исключениях
    session = requests.Session()
    session.trust_env = False
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=url, session=session)))
    trace.set_tracer_provider(provider)
    LangChainInstrumentor().instrument(tracer_provider=provider)
    _tracer = trace.get_tracer("rag")
    import atexit

    atexit.register(provider.shutdown)  # дослать спаны при выходе из CLI
    return True


@contextmanager
def llm_span(name: str, model: str, messages: list[dict], params: dict):
    """Спан вызова Ollama в терминах OpenInference (kind=LLM). Без трейсинга — пустой контекст.
    Вызывающий заполняет результат через record_llm_output(span, ...)."""
    if _tracer is None:
        yield None
        return
    from openinference.instrumentation.langchain import get_current_span
    from opentelemetry.trace import Status, StatusCode, set_span_in_context

    # Инструментирование LangChain не делает свои спаны текущими в контексте OTel: без явного родителя
    # вызов Ollama из узла графа попадает в отдельный трейс. Родитель — спан текущего узла LangGraph.
    # Вне графа (инструмент search) родителя из LangChain нет — тогда текущий контекст OTel (спан MCP tools/call).
    # get_current_span обращается к langchain_core.runnables, не импортируя его: без импорта — AttributeError.
    try:
        import langchain_core.runnables  # noqa: F401

        parent = get_current_span()
    except Exception:  # трейсинг не должен ронять запрос
        parent = None
    ctx = set_span_in_context(parent) if parent is not None else None
    with _tracer.start_as_current_span(name, context=ctx) as span:
        span.set_attribute("openinference.span.kind", "LLM")
        span.set_attribute("llm.system", "ollama")
        span.set_attribute("llm.model_name", model)
        span.set_attribute("llm.invocation_parameters", json.dumps(params, ensure_ascii=False))
        for i, m in enumerate(messages):
            span.set_attribute(f"llm.input_messages.{i}.message.role", m["role"])
            span.set_attribute(f"llm.input_messages.{i}.message.content", m["content"])
        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


def record_llm_output(span, content: str, data: dict) -> None:
    if span is None:
        return
    span.set_attribute("llm.output_messages.0.message.role", "assistant")
    span.set_attribute("llm.output_messages.0.message.content", content)
    span.set_attribute("output.value", content)
    pt, ct = data.get("prompt_eval_count"), data.get("eval_count")
    if pt is not None:
        span.set_attribute("llm.token_count.prompt", pt)
    if ct is not None:
        span.set_attribute("llm.token_count.completion", ct)
    if pt is not None and ct is not None:
        span.set_attribute("llm.token_count.total", pt + ct)
    # разбивка времени со стороны Ollama (нс -> с): всего на сервере, загрузка модели, prefill, генерация;
    # разница между длительностью спана и total_duration — сеть/HTTP
    for key in ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration"):
        if data.get(key) is not None:
            span.set_attribute(f"ollama.{key}_s", round(data[key] / 1e9, 3))
