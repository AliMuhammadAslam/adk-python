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

import pytest


def pytest_collection_modifyitems(items):
  """Skip all streaming/live tests.

  The workflow-based LlmAgent does not yet implement _run_live_impl.
  These tests will be re-enabled once live streaming support is added.
  """
  skip_marker = pytest.mark.skip(
      reason='Live streaming not yet supported by workflow-based LlmAgent'
  )
  for item in items:
    if 'streaming' in str(item.fspath):
      item.add_marker(skip_marker)
