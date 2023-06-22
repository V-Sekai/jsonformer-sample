# Copyright (c) 2018-present. This file is part of V-Sekai https://v-sekai.org/.
# SaracenOne & K. S. Ernest (Fire) Lee & Lyuma & MMMaellon & Contributors
# utils.py
# SPDX-License-Identifier: MIT

import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes
import threading
import time


def setup_logger():
    logger = logging.getLogger('custom_logger')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('output.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def setup_tracer():
    tracer = trace.get_tracer(__name__)
    resource = Resource.create({ResourceAttributes.SERVICE_NAME: "service_vtuber_generator"})
    trace.set_tracer_provider(TracerProvider(resource=resource))

    span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
    trace.get_tracer_provider().add_span_processor(span_processor)
    return tracer


def send_heartbeat(logger, tracer):
    with tracer.start_as_current_span("heartbeat") as heartbeat_span:
        while True:
            heartbeat_span.add_event("heartbeat")
            logger.info("Heartbeat event sent")
            time.sleep(20)


def start_heartbeat_thread(logger, tracer):
    heartbeat_thread = threading.Thread(target=send_heartbeat, args=(logger, tracer))
    heartbeat_thread.daemon = True
    heartbeat_thread.start()
