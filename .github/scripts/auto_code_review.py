# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

import google.generativeai as genai
import requests
from github import Github


model_name = os.environ["GOOGLE_MODEL_ID"]
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
max_context_token = 100000
force_review = False


def get_pr_diff():
    api_url = f"https://api.github.com/repos/{os.getenv('GITHUB_REPOSITORY')}/pulls/{os.getenv('PR_NUMBER')}"
    headers = {
        "Authorization": f"token {os.getenv('GITHUB_TOKEN')}",
        "Accept": "application/vnd.github.v3.diff",
    }
    response = requests.get(api_url, headers=headers)
    return response.text if response.status_code == 200 else ""


def get_prompt(diff, pr):
    system_prompt = """You are an experienced software engineer specializing in code reviews for deep learning libraries. Your task is to review code changes and related pull request (PR) information for `optimum-rbln`, a Python library that optimizes Hugging Face models for execution on RBLN NPUs.

Focus on providing actionable and constructive feedback. Don't make generalized suggestions."""

    prompt = f"""
Review the following code changes(GIT DIFF) along with the pull request (PR) details and provide feedback:

<PR_DESCRIPTION>
  title : {pr.title}
  body :
{pr.body[: pr.body.find("## Related Issues")] if pr.body is not None else ""}
</PR_DESCRIPTION>


<GIT_DIFF>
{diff}
</GIT_DIFF>
"""
    return system_prompt, prompt


def translate_review(review):
    model = genai.GenerativeModel(
        model_name,
        system_instruction="You are a professional translator specializing in technical and software-related content. Keep the technical words in English, but understand the whole sentence and rephrase it in Korean.",
    )
    prompt = f"""Translate the following English text into Korean, maintaining technical accuracy and clarity. Include ONLY the translation, NO OTHER EXPLANATIONS or RESPONSES as a chatbot. :

{review}"""
    response = model.generate_content(prompt)

    return response.text


def review_code(system_prompt, prompt):
    model = genai.GenerativeModel(model_name, system_instruction=system_prompt)
    response = model.generate_content(prompt)
    print(prompt)
    return response.text


def remove_file_from_diff(diff_content, file_to_remove):
    lines = diff_content.splitlines()
    result = []
    skip = False
    file_header = f"diff --git a/{file_to_remove} b/{file_to_remove}"

    for line in lines:
        if line.startswith("diff --git"):
            if line == file_header:
                skip = True
            else:
                skip = False

        if not skip:
            result.append(line)

    return "\n".join(result)


def skip_bot(pr):
    global force_review
    """
    Review if
    1. last commit messages starts with "[autoreview]"
    2. last comment contains "/autoreview"
    """

    # Check commit message
    commits = list(pr.get_commits())
    if len(commits) == 0:
        return True

    last_commit = commits[-1]
    try:
        commit_message = last_commit.raw_data["commit"]["message"]
    except KeyError:
        commit_message = ""

    if commit_message.startswith("[autoreview]"):
        return False

    # Check the last comment
    comments = list(pr.get_issue_comments())
    if len(comments) == 0:
        return True

    last = comments[-1]
    if last.user.login.find("github-actions") != -1:
        return True

    if last.body.find("/autoreview") == -1:
        return True

    if last.reactions["heart"] > 0:
        return True

    if last.body.find("force") != -1:
        force_review = True

    last.create_reaction("heart")
    return False


def main():
    github_token = os.getenv("GITHUB_TOKEN")
    pr_number = os.getenv("PR_NUMBER")
    if not pr_number:
        pr_number = os.getenv("INPUT_PR_NUMBER")

    if not all([github_token, pr_number]):
        print("Missing required environment variables")
        sys.exit(1)

    g = Github(github_token)
    repo = g.get_repo(os.getenv("GITHUB_REPOSITORY"))
    pr = repo.get_pull(int(pr_number))

    if skip_bot(pr):
        print(
            "To invoke review, Write '/autoreview' and re-run github actions,"
            " or start the commit message with '[autoreview]'. "
        )
        sys.exit(0)

    # Get PR diff
    diff = get_pr_diff()
    diff = remove_file_from_diff(diff, "uv.lock")

    # Check diff is available
    if len(diff) == 0:
        print("Failed to get the contents of PR Diff. Skipping review.")
        pr.create_issue_comment("Auto Code Review skipped: Failed to get the diff.")
        sys.exit(0)

    # check token count
    system_prompt, prompt = get_prompt(diff, pr)
    model = genai.GenerativeModel(model_name=model_name, system_instruction=system_prompt)
    num_tokens = model.count_tokens(prompt).total_tokens
    if num_tokens > max_context_token and not force_review:
        msg = f"Diff ({len(diff)}) exceeds maximum allowed tokens ({max_context_token}) > ({num_tokens}). Skipping review."
        print(msg)
        pr.create_issue_comment(msg)
        sys.exit(0)

    # Get Auto review
    review = review_code(system_prompt, prompt)
    translation = translate_review(review)

    # Post comment on PR
    pr.create_issue_comment(f"""# Auto Code Review

- [참고] Auto Code Review를 invoke하려면, commit message의 시작을 [autoreview]로 시작하거나, "/autoreview" 를 comment로 작성한 후,
해당 commit의 github action에서 code review를 re-run 하시면 됩니다.
\n\n{review}\n\n{translation}""")


if __name__ == "__main__":
    main()
