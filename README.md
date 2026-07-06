# Trustworthy Language Model (TLM)

The [Trustworthy Language Model](https://cleanlab.ai/blog/trustworthy-language-model/) scores the **trustworthiness** of outputs from *any* LLM in *real-time*.

Automatically detect hallucinated/incorrect responses in: Q&A (RAG), Chatbots, Agents, Structured Outputs, Data Extraction, Tool Calling, Classification/Tagging, Data Labeling, and other LLM applications.

Use TLM to:
- Guardrail AI mistakes before they are served to user
- Escalate cases where AI is untrustworthy to humans
- Discover incorrect LLM (or human) generated outputs in datasets/logs
- Boost AI accuracy

Powered by *uncertainty estimation* techniques, TLM **works out of the box**, and does **not** require: <br>
data preparation/labeling work or custom model training/serving infrastructure.


## Resources

- [Documentation and Tutorials](https://cleanlab.github.io/tlm/)
- [Example Notebooks](notebooks)
- Learn more and see precision/recall benchmarks with frontier models (from OpenAI, Anthropic, Google, etc): <br>
[Blog](https://cleanlab.ai/blog/), [Research Paper](https://aclanthology.org/2024.acl-long.283/)


This code implements a more effective variant of the BSDetector LLM uncertainty-quantification method from our paper:

```bibtex
@inproceedings{bsdetector,
  title={Quantifying uncertainty in answers from any language model and enhancing their trustworthiness},
  author={Chen, Jiuhai and Mueller, Jonas},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={5186--5200},
  year={2024}
}
```
