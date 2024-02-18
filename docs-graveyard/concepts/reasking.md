# How does reasking work

## This is auto generated -- don't commit as is.

Reasking is a feature of Guardrails that allows you to re-ask the model for a response if the initial response does not meet the criteria defined in the RAIL spec.  This is useful when you want to ensure that the model's response meets certain criteria, such as being on-topic, being factually accurate, or being sufficiently different from the input prompt.

When you use reasking, Guardrails will automatically re-ask the model for a response up to a certain number of times, until the response meets the criteria defined in the RAIL spec.  You can specify the number of re-asks using the `num_reasks` parameter when calling the `Guard` object.