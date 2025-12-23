---
title: Cheat sheet / FAQ for the Trustworthy Language Model
sidebar_label: Cheat sheet / FAQ
---

# Cheat sheet / FAQ for TLM

Make sure you've read the [quickstart tutorial](tutorials/quickstart/index.ipynb) and [advanced tutorial](tutorials/tlm_advanced/index.ipynb).

## Tips on using TLM for specific tasks

Recall the two ways to use TLM:

1. Generate responses via your own LLM, and then use TLM to score their trustworthiness via TLM's [`get_trustworthiness_score()`](tutorials/quickstart/index.ipynb#scoring-the-trustworthiness-of-a-given-response) method.
2. Use TLM in place of your LLM to both generate responses and score their trustworthiness via TLM's [`prompt()`](tutorials/quickstart/index.ipynb#using-tlm) method.

Choose Option 1 to:

- Stream in responses at the lowest latency, and then score their trustworthiness.
- Use a specific LLM model not supported within TLM, or keep existing LLM inference code as is.

Choose Option 2 to:

- Simply use one API call to produce both responses and trust scores.
- Auto-improve LLM responses (achieved by increasing the `num_candidate_responses` configuration in [TLMOptions](api/python/tlm/#class-tlmoptions)).

In either case, ensure TLM receives the **same information/prompt** you'd supply to your own LLM.
More tips for particular AI applications:

<details id="rag">
<summary>Retrieval-Augmented Generation (RAG)</summary>

Refer to our [RAG tutorial](use-cases/tlm_rag/index.ipynb). For RAG, we recommend TrustworthyRAG over the standard TLM object.

Common mistakes when using TLM for RAG include:

- The `prompt` provided to TLM is missing: the retrieved **Context**, or your **System Instructions** (e.g. when to abstain / say 'No information available', specific requirements for a correct/good response). Provide TLM with the _same_ `prompt` you'd use for your own LLM in RAG.
- Running TLM at default settings if you require low-latency. Try the lower-latency TLM configurations listed on this page.

In RAG: what makes TLM different than groundedness/faithfulness scores like RAGAS/DeepEval?

While some developers measure many scores to debug RAG system components, what matters most to your users is: did the RAG system answer correctly or not. TLM trustworthiness scores detect incorrect RAG responses in real-time with 3x greater precision than groundedness/faithfulness scores like RAGAS (see [benchmarks](https://cleanlab.ai/blog/rag-tlm-hallucination-benchmarking/)). Groundedness/faithfulness measures like RAGAS only attempt to estimate discrepancies between the RAG response and retrieved Context, and thus only detect certain response errors. TLM relies on state-of-the-art model uncertainty estimation, which detects the same discrepancies but also issues such as when: the response is not a good answer for the user's query (LLMs often make reasoning/factual errors), or the query is complex/vague, or the retrieved Context is confusing or not relevant/sufficient for a proper answer.

Beyond TLM's built-in score for response trustworthiness, you can also use our [TrustworthyRAG Evals](use-cases/tlm_rag/index.ipynb) to simulateneously score: groundedness/faithfulness, abstention, context sufficiency, response helpfulness, query difficulty, or custom properties of your RAG system -- all in a more efficient and reliable manner than tools like RAGAS or DeepEval.

</details>

<details id="agents">
<summary>Agents</summary>

For adding trust scoring into a prebuilt LangGraph Agent, see this [tutorial](use-cases/tlm_existing_agent/index.ipynb).

For adding trust scoring into a customized Agent, see this [tutorial](use-cases/tlm_agents/index.ipynb). It's also focused on LangGraph, since that is the most flexible framework for customized Agents, but TLM can be used in _any_ Agentic framework (OpenAI Agents SDK, Crew AI, ...).

Two ways we recommend to add trust scoring for your Agent:

1. scoring the trustworthiness of the final LLM output (i.e. Agent's response to user)
2. scoring the trustworthiness of _every_ LLM output, including internal LLM calls made by the Agent.

The latter may be useful to prevent: your Agent from 'going off the rails', or bad Tool Calls from happening (e.g. can escalate to human approval before any Tool Call with low trustworthiness score is executed).

When LLM outputs receive low trustworthiness scores, you can consider several fallback strategies:

1. [Replace Agent's responses with a pre-written abstention phrase indicating lack of knowledge](use-cases/tlm_agents/index.ipynb).
2. Escalate this Agent interaction to a human.
3. Automatically handle it within the Agent to autonomously improve response accuracy -- either by: re-running the Agent with a modified prompt, or only [re-generating the recent untrustworthy LLM output](use-cases/tlm_existing_agent/index.ipynb).

</details>

<details id="summarization">
<summary>Summarization</summary>

Include specific instructions in your [prompt](https://www.promptingguide.ai/prompts/text-summarization), such as the desired length of the summary, format, and what types of information/concepts are most/least important to include.

</details>

<details id="chat">
<summary>Conversational Chat (handling system prompts and message history)</summary>

Refer to our [multi-turn conversations tutorial](use-cases/tlm_conversation/index.ipynb).

For chatbots: TLM's trustworthiness scoring can be useful for automated escalation to a human agent, or to flag key responses as potentially untrustworthy to your users.

TLM remains effective when system prompts and past message history are included in its `prompt` argument in various formats. For example, you could set TLM's `prompt` to the following string (which implies the next answer will come from the AI):

```
System: You are a customer support agent representing company XYZ.

User: hi

Assistant: How can I help you?

User: can I return my earrings?

Assistant:
```

This is also how packages like LangChain handle conversation history.

You can alternatively use OpenAI's conversation history and system prompt handling, by [running TLM via the OpenAI API](tutorials/tlm_structured_outputs/index.ipynb).

</details>

<details id="extraction">
<summary>Data Extraction</summary>

- Refer to our [data extraction tutorial](use-cases/tlm_data_extraction/index.ipynb#convert-pdf-documents-to-text).

- The TLM trustworthiness score tells you which data auto-extracted from documents, databases, transcripts is worth double checking to ensure accuracy. Consider running TLM with a higher number of `num_candidate_responses` (specified in [TLMOptions](api/python/tlm/#class-tlmoptions)) to auto-improve LLM accuracy in addition to scoring LLM trustworthiness.

- If you already know which section in your documents contains the relevant information, save cost and boost accuracy by only including text from the relevant part of the document in your prompt.

- If you wish to extract multiple structured data fields from each unstructured document, consider using [Structured Outputs](tutorials/tlm_structured_outputs/index.ipynb).

</details>

<details id="classification">
<summary>Classification</summary>

- Refer to our [Zero-Shot Classification tutorial](use-cases/zero_shot_classification/index.ipynb). Or for binary classification, our [Yes/No Decisions tutorial](use-cases/tlm_yes_no_decision/index.ipynb).

- Pass in the `constrain_outputs` keyword argument to `TLM.prompt()` to restrict the output to your set of classes/categories.

- Consider running TLM with a higher number of `num_candidate_responses` (specified in [TLMOptions](api/python/tlm/#class-tlmoptions)) to auto-improve LLM accuracy in addition to scoring LLM trustworthiness.

- A good prompt template should list all of the possible categories a document/text can be classified as, definitions of the categories, and instructions for the LLM to choose a category (including how to handle edge-cases). Append this template to the text of each document in order to form the `prompt` argument for TLM. After running TLM, you can review the most/least trustworthy LLM predictions and then refine your prompt based on this review.

- If you have some already labeled examples from different classes, try [few-shot prompting](https://www.promptingguide.ai/techniques/fewshot), where these examples and their classes are listed within the prompt template.

- You can also try using [Structured Outputs](tutorials/tlm_structured_outputs/index.ipynb), although today's LLMs display lower accuracy in some classification/tagging tasks when required to structure their outputs.

</details>

<details id="annotation">
<summary>Data Annotation/Labeling</summary>

- Refer to our [Data Annotation tutorial](use-cases/tlm_annotation/index.ipynb). Also check out the various tips/tutorials on using TLM for classification, structured outputs, and data extraction -- these cover ideas useful for data annotation as well.

- LLMs (including TLM) can handle most types of data labeling (including: text categorization, document tagging, entity recognition / PII detection, and more complex annotation tasks). The TLM trustworthiness scores additionally reveal which subset of data the LLM can confidently handle. Let the LLM auto-label 99% of cases where it is trustworthy, and manually label the remaining 1%.

- Consider running TLM with a higher number of `num_candidate_responses` (specified in [TLMOptions](api/python/tlm/#class-tlmoptions)) to auto-improve LLM accuracy in addition to scoring LLM trustworthiness.

- TLM can also detect labeling errors made by human annotators (examples where TLM confidently assigns a different label than the human annotator).

- Provide detailed annotation instructions and example annotations in TLM's `prompt` argument. At least the same level of detail as the human annotator instructions (preferably more detail since LLMs can quickly process more information than humans).

</details>

<details id="nonstandard-outputs">
<summary>Non-Standard Response Types: Structured Outputs, Function Calling, ...</summary>

Currently, you should use [TLM via the OpenAI API](tutorials/tlm_structured_outputs/index.ipynb) to handle non-standard output types. Used this way, TLM can score the trustworthiness of _every_ type of output that OpenAI can return, including Structured Outputs, Tool Calls, etc.

</details>

<details id="non-english">
<summary>Non-English Responses: Other Languages, Code Generation, ...</summary>

For code generation applications, specify the TLM `task` for better results:

```
tlm = TLM(task="code_generation")
```

TLM can generally be used in any LLM application where the base LLM model is at least moderately capable.
This includes applications where your LLM outputs: a foreign language, code/SQL, or other types of symbolic strings (e.g. molecular formula, board game encoding, ...).

For such applications: run TLM with the base [`model`](api/python/tlm/#class-tlmoptions) that works best for your use-case.
TLM's default settings are optimized for English generation, so you may achieve better results with one of the following [TLMOptions configurations](api/python/tlm/#class-tlmoptions):

```
options = {"reasoning_effort": "none", "similarity_measure": "string"}

options = {"reasoning_effort": "none", "similarity_measure": "discrepancy"}

options = {"reasoning_effort": "none", "similarity_measure": "embedding"}  # or "embedding_large"

options = {"num_consistency_samples": 0}
```

</details>

<details id="evals">
<summary>LLM Evals, or improving LLM fine-tuning</summary>

For LLM Evals, use TLM to quickly find bad/untrustworthy LLM responses in your application logs. Inspecting the least trustworthy LLM responses helps you discover how to improve your prompts/model (e.g. how to handle edge-cases).

For improving LLM fine-tuning, use TLM to find bad training data and then filter/correct it.

The following tutorials can help:

- [Auto-detect bad responses in a dataset](tutorials/quickstart/index.ipynb#application-scoring-the-trustworthiness-of-pre-generated-responses)
- [Account for custom evaluation criteria and human feedback](tutorials/tlm_custom_eval/index.ipynb)
- [Explain why responses are flagged as untrustworthy/low-quality](tutorials/tlm_advanced/index.ipynb#explaining-low-trustworthiness-scores)
- [Auto-detect more types of bad data/responses](use-cases/instruction_tuning_data/index.ipynb)

</details>

## Recommended TLM configurations to try

TLM offers [optional configurations](tutorials/tlm_advanced/index.ipynb#quality-presets). The default TLM configuration is not latency/cost-optimized because it must remain effective across all possible LLM use-cases. For your specific use-case, you can greatly improve latency/cost without compromising results. Strategy: first run TLM with default settings to see what results look like over a dataset from your use-case; once results look promising, adjust the TLM preset/[options/model](api/python/tlm/#class-tlmoptions) to reduce latency for your application. If TLM's default configuration seems ineffective, switch to a more powerful [`model`](api/python/tlm/#class-tlmoptions) or add [custom evaluation criteria](tutorials/tlm_custom_eval/index.ipynb).

We list some good configurations to try out below. Each can be copy/pasted into the initialization arguments for the TLM object:

```
tlm = TLM(<configuration>)
```

#### For low latency (real-time applications):

```
quality_preset = "base"
```

or:

```
quality_preset = "low", options = {"model": "gpt-4.1-nano"}  # consider "base" instead of "low", "gpt-5-nano" or "nova-micro" instead of "gpt-4.1-nano"
```

#### For better trustworthiness scoring:

```
options = {"model": "gpt-5"}  # or consider: "o4-mini", "gpt-4.1"
```

#### For more accurate LLM responses:

```
options = {"model": "o4-mini", "num_candidate_responses": 4}  # higher number of `num_candidate_responses` could produce more accurate responses
# Or instead of "o4-mini", consider: "o3", "gpt-5", or "claude-sonnet-4-0"
```

## Frequently Asked Questions

<details id="how-does-it-work">

<summary>How does the TLM trustworthiness score work?</summary>

TLM significantly improves the quality of LLM responses by reducing mistakes and outperforming other evaluation methods.

In extensive benchmarks, TLM detects hallucinations and incorrect answers with much higher precision and recall than alternatives like logprobs, LLM-as-judge, RAGAS, DeepEval, and others. These results hold across a wide range of datasets, LLMs, and customer support tasks.

With using the `best` or `high` [quality_preset](api/python/tlm/#class-tlmoptions), TLM doesn't just evaluate, it helps improve outputs. For example, across a sample set of models, it reduced the error rate of LLM responses by:

- 27% for GPT-4o
- 34% for GPT-4o mini
- 22% for GPT-3.5
- 10% for GPT-4
- 24% for Claude 3 Haiku

You can explore the results in our published benchmarks:

- [TLM: Trustworthy Language Model](https://cleanlab.ai/blog/trustworthy-language-model)
- [RAG Evaluation Models Benchmark](https://cleanlab.ai/blog/rag-evaluation-models)
- [RAG Hallucination Metrics Benchmark](https://cleanlab.ai/blog/rag-tlm-hallucination-benchmarking)
- [Get in touch](https://cleanlab.ai/contact/) to learn more or see how TLM can help your use case.

</details>

<details id="trust-trust-scores">

<summary>Why should I trust the TLM trustworthiness scores?</summary>

For transparency and scientific rigor, we [published](https://aclanthology.org/2024.acl-long.283/) our state-of-the-art research behind TLM in ACL, the top venue for NLP and Generative AI research. TLM combines all major forms of uncertainty quantification and LLM-based evaluation into one unified framework that comprehensively detects different types of LLM mistakes.

Ultimately what matters is whether TLM actually detects LLM errors in real applications. Rigorous [benchmarks](https://cleanlab.ai/blog/trustworthy-language-model/) reveal that TLM trustworthiness scores detect wrong responses with _significantly greater precision_ than alternative approaches like: token probabilities (_logprobs_), or asking the LLM to directly evaluate the response (_LLM-as-judge_). Such findings hold across diverse use-cases, domains, and all major LLMs [including reasoning models](https://cleanlab.ai/blog/tlm-o1/). In extensive [RAG benchmarks](https://towardsdatascience.com/benchmarking-hallucination-detection-methods-in-rag-6a03c555f063/), TLM detected incorrect RAG responses with _significantly greater precision_ than alternatives including: RAGAS, LLM-as-judge, G-Eval, DeepEval, HHEM, Lynx, Prometheus-2, or LogProbs.

Additional accuracy [benchmarks](https://cleanlab.ai/blog/llm-accuracy/) reveal that TLM's trustworthiness score can be used to automatically improve LLM responses themselves (in the same way across many LLM models). This would not be possible if the trustworthiness score were unable to automatically catch incorrect LLM responses.

</details>

<details id="why-low-trust">

<summary>Why did TLM return a low trustworthiness score?</summary>

Our [Advanced Tutorial](tutorials/tlm_advanced/index.ipynb#explaining-low-trustworthiness-scores) demonstrates how to activate _explanations_ and understand why a particular response is considered untrustworthy.

We typically consider trustworthiness scores below 0.7 **low**, and scores above 0.9 **high**. But it depends on use-case, especially for scores between 0.7 - 0.9.

Remember that TLM estimates the _uncertainty_ in your LLM. Trustworthiness scores may be low even for correct LLM responses, if your LLM would likely get similar questions wrong. Trustworthiness scores may also be low for open-ended requests for which there isn't a single correct answer (see one way to handle this [here](faq.md#chat)).

Simple configurations like a more powerful [`model`](api/python/tlm/#class-tlmoptions) can improve the trustworthiness scores for your use-case; here are [tips](tutorials/tlm_advanced/index.ipynb#optional-tlm-configurations-for-betterfaster-results).

</details>

<details id="align-trust-scores">

<summary>Why don't TLM trust scores align with my team's human evaluations of LLM outputs?</summary>

Our [Custom Evaluation Criteria tutorial](tutorials/tlm_custom_eval/index.ipynb) demonstrates how to better tailor TLM for response quality ratings specific to your use-case.

Simple configurations like a more powerful [`model`](api/python/tlm/#class-tlmoptions) can improve the trustworthiness scores for your use-case; here are [tips](tutorials/tlm_advanced/index.ipynb#optional-tlm-configurations-for-betterfaster-results).

</details>

<details id="scores-match">

<summary>Why don't trustworthiness scores from `TLM.prompt()` and `TLM.get_trustworthiness_score()` always match?</summary>

The scores are also not deterministic and are computed as a result of multiple (non-deterministic) LLM calls.
When re-running TLM on the same prompt, results are cached and thus you may get identical results until the cache is refreshed. The cache automatically refreshes every 30 days or sooner in rare cases such as major updates or internal refreshes.

`TLM.prompt()` additionally considers statistics produced during LLM response generation (such as token probabilities), whereas `TLM.get_trustworthiness_score()` does not.

If you want to use one base LLM model to generate responses and score their trustworthiness with a different (e.g. faster) base LLM model, you can still obtain the `.prompt()` trustworthiness score via [TLM Lite](tutorials/tlm_advanced/index.ipynb#trustworthy-language-model-lite).

</details>

<details id="reduce-latency">

<summary>How can I reduce latency or costs when using TLM?</summary>

By default, TLM is configured for broad reliability across all LLM use cases, not for speed or cost-efficiency. But you can significantly reduce latency and runtime costs by customizing settings for your specific application without sacrificing accuracy.

### Recommended Approach

1. Start with default settings to evaluate TLM on a sample from your use case.
1. Once results look solid, adjust the configuration to optimize for latency or cost.

### Ways to Reduce Latency and Cost

- Lower the [quality preset](tutorials/tlm_advanced/index.ipynb#quality-presets): Use `quality_preset="low"` or `"base"` to get faster and cheaper results while still catching most LLM errors.
- Use a faster model: In [TLMOptions](api/python/tlm/#class-tlmoptions), choose lightweight models like `gpt-5-nano`, `gpt-4.1-nano`, or `nova-micro`.
- Reduce reasoning complexity: Set `reasoning_effort="none"` and `similarity_measure="string"` to cut down processing time.
- Use [TLM Lite](tutorials/tlm_advanced/index.ipynb#trustworthy-language-model-lite): This variant provides quicker trustworthiness scores and pairs well with high-quality but slower LLM responses.
- Stream scores separately: Stream in responses from your own LLM, then use `TLM.get_trustworthiness_score()` to stream in the trust score afterward — enabling near real-time trust evaluation.

### For Enterprise Use

If latency is critical, you can deploy TLM privately in your own VPC. This reduces network latency and leverages your existing LLM infrastructure. TLM doesn’t require any additional infrastructure as it runs on top of your current setup.

[Reach out](https://cleanlab.ai/contact/) to learn more.

</details>

<details id="costs">
<summary>How much does TLM cost?</summary>

You can try TLM for free! Sign up for a Cleanlab account [here](https://tlm.cleanlab.ai) to get your API key, and have fun trying TLM in your LLM workflows.

Once your free tokens are used up, you can continue using this same TLM API on a pay-per-token plan. View the pricing in your [Cleanlab Account](https://tlm.cleanlab.ai/account) under `Usage & Billing`. TLM offers many base LLM models and configuration settings like [quality presets](tutorials/tlm_advanced/index.ipynb#quality-presets), giving you flexible pricing options to suit your needs.

The default TLM settings are more expensive because they have to remain effective across all possible LLM use-cases. For your specific use-case, you can greatly improve costs without compromising results. Strategy: first run TLM with default settings to see what results look like over a dataset from your use-case; once results look promising, adjust the TLM preset/options/model to reduce costs for your application. For instance, you can reduce costs significantly via [TLM Lite](tutorials/tlm_advanced/index.ipynb#trustworthy-language-model-lite).

Enterprise subscriptions are available with: volume discounts, private deployment options, and many additional features. [Reach out](https://cleanlab.ai/sales/) to learn more.

</details>

<details id="big-dataset">
<summary>How to run TLM over big datasets?</summary>

Refer to our [Advanced Tutorial](tutorials/tlm_advanced/index.ipynb#running-tlm-over-large-datasets).

Hitting your account's rate limits may cause variable speeds when using TLM to process a dataset.

Contact us if you are still having problems: support@cleanlab.ai

</details>

<details id="prompt-to-use">
<summary>What prompt should I use for TLM?</summary>

When specifying TLM's `prompt` argument, you should **not** include instructions to evaluate responses or utilize LLM-as-a-judge style prompting.
For instance, you do **not** need to say something like: _"Evaluate the trustworthiness of this response"_.

Instead, run TLM with the _same_ `prompt` you'd use to generate a response with your own LLM. This makes TLM easy to include in any LLM application, and produces _better_ trustworthiness scores. You just focus on prompts to produce good responses from your AI application, and let TLM handle trust scoring.

</details>

<details id="vpc">
<summary>Do you offer private deployments in VPC?</summary>

Yes, TLM can be deployed in your company's own cloud such that all data remains within your private infrastructure. All major cloud providers and LLM models are supported. [Reach out](https://cleanlab.ai/sales/) to learn more.

</details>

<details id="other-llm">
<summary>My company only uses a proprietary LLM, or a specific LLM provider</summary>

You can use `TLM.get_trustworthiness_score()` to score the trustworthiness of responses from _any_ LLM. See our tutorial: [Compute Trustworthiness Scores for any LLM](tutorials/tlm_custom_model/index.ipynb)

If you would like to both produce responses and score their trustworthiness using your own custom (private) LLM, [get in touch](https://cleanlab.ai/contact/) regarding our Enterprise plan. Our TLM technology is compatible with _any_ LLM or Agentic system.

</details>

<details id="how-use-tlm">
<summary>I am using LLM model ___, how can I use TLM?</summary>

Two primary ways to use TLM are the [prompt()](https://web.archive.org/web/20250518103618/https://help.cleanlab.ai/tlm/) and [get_trustworthiness_score()](https://web.archive.org/web/20250518103618/https://help.cleanlab.ai/tlm/) methods. The former can be used as a drop-in replacement for any standard LLM API, returning trustworthiness scores in addition to responses from one of TLM’s supported base LLM models. Here the response and trustworthiness score are both produced using the same LLM model.

Alternatively, you can produce responses using [any LLM](tutorials/tlm_custom_model/index.ipynb), and just use TLM to subsequently score their trustworthiness.

If you would like to both produce responses and score their trustworthiness using your own custom (private) LLM, [get in touch](https://cleanlab.ai/contact/) regarding our Enterprise plan.

</details>

## Learn More

Beyond the tutorials in this documentation and tips on this page, you can learn more about TLM via our [blog](https://cleanlab.ai/blog/?tag=TLM) and additional [cookbooks](https://github.com/cleanlab/cleanlab-tools/). For instance, the [TLM demo cookbook](https://github.com/cleanlab/cleanlab-tools/blob/main/TLM-Demo-Notebook/TLM-Demo.ipynb) provides a concise demo of TLM used across various applications (particularly customer support use-cases).

If your question is not answered here, feel free to ask in our [Community Slack](https://cleanlab.ai/slack), or via email: [support@cleanlab.ai](mailto:support@cleanlab.ai)
