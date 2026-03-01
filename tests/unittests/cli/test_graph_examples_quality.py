# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Quality checks for GraphAgent example scripts."""

from __future__ import annotations

import importlib
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[3]


def test_run_all_examples_script_fails_on_example_errors():
  script = (
      _ROOT
      / "contributing"
      / "samples"
      / "graph_examples"
      / "run_all_examples.sh"
  ).read_text(encoding="utf-8")

  assert "set -euo pipefail" in script
  assert "failed_examples" in script
  assert "|| true" not in script
  assert ".venv/bin/activate" in script


def test_parallel_checkpointing_result_helpers():
  module = importlib.import_module(
      "contributing.samples.graph_examples.12_parallel_checkpointing.agent"
  )

  text_a = "Worker 'worker_a': result_a='processed_by_a'"
  text_b = "Worker 'worker_b': result_b='processed_by_b'"

  assert module._extract_result_value(text_a, "a") == "processed_by_a"
  assert module._extract_result_value(text_b, "b") == "processed_by_b"
  assert module._extract_result_value("no result here", "a") is None

  session_state = {}
  graph_data = {
      "worker_a": text_a,
      "worker_b": text_b,
  }
  events = []
  assert module._get_result_value(session_state, graph_data, "a", events) == (
      "processed_by_a"
  )
  assert module._get_result_value(session_state, graph_data, "b", events) == (
      "processed_by_b"
  )

  session_state = {"result_a": "from_session"}
  assert module._get_result_value(session_state, graph_data, "a", events) == (
      "from_session"
  )
