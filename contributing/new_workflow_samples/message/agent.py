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

import asyncio
import base64
from typing import Any

from google.adk import Event
from google.adk.workflow._workflow_class import Workflow
from google.genai import types


def send_string(node_input: Any = None):
  """Sends a single string message."""
  yield Event(message="1. This is a simple string message.")


def send_multimodal(node_input: Any = None):
  """Sends a multi-modal message containing a string and an inline image."""
  # A 16x16 solid red PNG base64 encoded
  red_square_png = base64.b64decode(
      "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAXElEQVR4nO2TSQ7AIAwD7fz/z+ZQtapwmrJc8QklmjBIgZJgIZMiAIl9KYbhjx4fgwosbNxgMrF0+4uhgHnYDM6AzQHJeg5HYtyHFfgy2AztN/5tZWfrBtVzkl4DzfQkEPd+cEkAAAAASUVORK5CYII="
  )
  yield Event(
      message=[
          types.Part.from_text(
              text=(
                  "2. Here is a multi-modal message with an inline image (red"
                  " square):"
              )
          ),
          types.Part.from_bytes(data=red_square_png, mime_type="image/png"),
      ]
  )


async def multiple_messages(node_input: Any = None):
  """Sends multiple complete messages from the same node with an interval."""
  yield Event(message="3. Processing step 1...")
  await asyncio.sleep(1.0)

  yield Event(message="3. Processing step 2...")
  await asyncio.sleep(1.0)

  yield Event(message="3. Done processing.")


root_agent = Workflow(
    name="message",
    edges=[
        (
            "START",
            send_string,
            send_multimodal,
            multiple_messages,
        ),
    ],
)
