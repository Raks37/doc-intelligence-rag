"""
Requirement: "JSON/XML schema mapping & Kafka."

Publishes pipeline results as events for downstream consumers (e.g. Contact
Master loader, exception-queue workflow tool, BOT integration). The Pydantic
schemas double as the event schema -- `.model_dump_json()` gives you a
versioned, validated JSON payload for free.

For the demo, KAFKA_ENABLED defaults to False so it runs without a live broker;
flip it on against a local Kafka (e.g. `docker run -p 9092:9092 apache/kafka`)
to show real publish/consume in the interview if you have time.
"""
import os
import json
import logging

logger = logging.getLogger("kafka_producer")
KAFKA_ENABLED = os.getenv("KAFKA_ENABLED", "false").lower() == "true"
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

TOPIC_EXTRACTION_COMPLETE = "doc-intelligence.extraction.completed"
TOPIC_EXCEPTIONS = "doc-intelligence.exceptions.raised"
TOPIC_COMPARISON_RESULT = "doc-intelligence.comparison.completed"

_producer = None


def _get_producer():
    global _producer
    if _producer is None and KAFKA_ENABLED:
        from kafka import KafkaProducer
        _producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
        )
    return _producer


def publish_event(topic: str, key: str, payload: dict) -> None:
    if not KAFKA_ENABLED:
        logger.info("[KAFKA-SIMULATED] topic=%s key=%s payload_keys=%s",
                     topic, key, list(payload.keys()))
        return
    producer = _get_producer()
    producer.send(topic, key=key.encode("utf-8"), value=payload)
    producer.flush()
    logger.info("[KAFKA] published to %s key=%s", topic, key)
