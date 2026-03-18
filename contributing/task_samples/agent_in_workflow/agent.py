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

from google.adk import Agent
from google.adk import Event
from google.adk import Workflow
from google.adk.workflow import DEFAULT_ROUTE
from pydantic import BaseModel
from pydantic import Field


class PatientIdentity(BaseModel):
  """Output schema for the intake agent."""

  name: str = Field(description="The patient's full name.")
  phone_number: str = Field(description="The patient's phone number.")


intake_agent = Agent(
    name="intake_agent",
    mode="task",
    output_schema=PatientIdentity,
    instruction="""\
You are a medical lab intake assistant. Your job is to chat with
the user to get their full name and phone number. Do not make up
information. Once you have both, finish your task.""",
)


def find_orders(node_input: PatientIdentity):
  """Mocks checking the database for the patient.

  Routes back to intake_agent if the name is not Jane Doe.
  Otherwise routes to success.
  """
  if node_input.name.lower() != "jane doe":
    yield Event(
        message=(
            f"Could not find matching records for {node_input.name}. Let's"
            " try again."
        ),
        route="retry",
    )
  else:
    yield Event(state={"orders": ["CBC (Complete Blood Count)", "Lipid Panel"]})


generate_instruction = Agent(
    name="generate_instruction",
    instruction="""
Based on the following orders, generate a concise instruction about how to prepare.
{orders}
""",
)


def send_message(orders: list[str], node_input: str):
  """Sends a final message with the found orders."""
  orders_str = "\n- ".join(orders)
  yield Event(message=f"""\
Good news! We found your orders:
- {orders_str}

Please follow these instructions:

{node_input}""")


root_agent = Workflow(
    name="task_in_workflow",
    edges=[
        ("START", intake_agent, find_orders),
        (
            find_orders,
            {"retry": intake_agent, DEFAULT_ROUTE: generate_instruction},
        ),
        (generate_instruction, send_message),
    ],
)
