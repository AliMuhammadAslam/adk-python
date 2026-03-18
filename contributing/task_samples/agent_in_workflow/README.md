# Agents In Workflow

This sample demonstrates how to use both `task` mode and `single_turn` mode Agents as nodes within a `Workflow`.

## Overview

The workflow represents a medical lab intake process:

1. **`intake_agent`**: A `task` mode Agent that chats with the user to collect their `name` and `phone_number`. It handles a multi-turn conversation until its `PatientIdentity` output schema is fulfilled.
1. **`find_orders`**: A regular Python function node that receives the `PatientIdentity`. It mocks a database lookup.
   - If the name is anything other than "Jane Doe", it yields a `retry` route, sending the user back to the `intake_agent`.
   - If the name is "Jane Doe", it places the lab orders into the state and yields to the `DEFAULT_ROUTE`.
1. **`generate_instruction`**: A `single_turn` mode Agent that automatically maps state parameters into its instruction template (`{orders}`) and generates a concise instruction about how to prepare.
1. **`send_message`**: A function node that reads both the `orders` from the state and the generated instructions from `node_input` to construct and send a final message to the user.

## Sample Inputs

- `Hi, I am Jane Doe, my phone number is 555-1234.`

  *The system will process this and return the mock lab orders along with AI-generated instructions on how to prepare.*

- `I'm here for my blood work.`

  *The system will ask for your name and phone number.*

- `My name is John Doe, and my number is 123-456-7890.`

  *The system will fail to find John's orders and route back to the intake agent.*

## Graph

```text
                  [ START ]
                      |
                      v
              [ intake_agent ] <----.
                      |             |
                      v             |
               [ find_orders ] --- retry
                      |
                      | (DEFAULT_ROUTE)
                      v
          [ generate_instruction ]
                      |
                      v
              [ send_message ]
```

## How To

Within an ADK workflow, you can embed LLM agents directly as nodes. The ADK runner handles them according to their `mode`:

### 1. Task Mode Agents

A `task` agent (`mode="task"`) handles a multi-turn conversation on its own before passing control to the next node. It will continually interact with the user until its specified task is completed.

```python
class PatientIdentity(BaseModel):
  name: str
  phone_number: str

intake_agent = Agent(
    name="intake_agent",
    mode="task", # Stops and chats with the user until the schema is populated
    output_schema=PatientIdentity,
    instruction="...",
)
```

The parsed `output_schema` object is automatically forwarded as the `node_input` to the next node in the graph.

### 2. Single Turn Mode Agents

A `single_turn` agent (the default mode if omitted) executes a single LLM call. It is typically used for inline text generation, summarization, or classification without chatting with the user.

```python
generate_instruction = Agent(
    name="generate_instruction",
    # mode="single_turn" is default in a workflow environment.
    instruction="Generate instructions for the following tests:\n{orders}",
)
```

In a workflow, single-turn agents send their `node_input` to the LLM model as content, and developers can inject state values into the instruction template. In this particular sample, the `generate_instruction` agent does not receive any explicit `node_input` from the prior node, so only the instruction (with the `{orders}` state value interpolated) is sent to the model. The resulting generated text is then forwarded to the next node as `node_input`.
