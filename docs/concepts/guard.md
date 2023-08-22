# Guard


The guard object is the main interface for GuardRails. It is seeded with a RailSpec, and then used to run the GuardRails AI engine. It is the object that accepts changing prompts, wraps LLM prompts, and keeps track of call history.


## How it works

```mermaid
graph
    A[Create `RAIL` spec] --> B[Initialize `guard` from spec];
    C[Collect LLM callable information] --> D[Invoke `guard`];
    B --> D;
    E[Invoke guard with prompt and instructions] --> D;
    D --> F[Guard invokes LLM API];
    F --> G[LLM API returns];
    G --> H[LLM metadata is stored and logged];
    H --> I[LLM output is validated];
    I --> J[Valid];
    I --> K[Invalid];
    K --> L[Check the on-fail action set in failed validator];
    L --> M[default: no-op];
    M --> N[Return output];
    L --> O[reask: reask LLM];
    O --> F;
    L --> P[filter: filter out output];
    P --> N;
    L --> Q[fix: fix output based on validator fix function]
    Q --> N;
    J --> N;
```