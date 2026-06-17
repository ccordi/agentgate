#!/usr/bin/env python3
"""Synthetic mixed-sensitivity corpus generator.

Generates a labeled, clearly-synthetic corpus of exactly 150 items:
  * 50 public items (benign OSS content, expected route: cloud, no action)
  * 50 sensitive_doc items (synthetic PII, expected route: local)
  * 50 secret_bearing items (config/code with a planted fake secret,
    expected route: cloud with redaction)

This generator is fully deterministic and self-validating.
"""

from __future__ import annotations

import random

from agentgate.security.classifier import Sensitivity, classify
from agentgate.security.redaction import detect

from ..loader import SENSITIVITY_CORPUS, write_jsonl
from ..schema import CorpusItem, LabelOrigin


def generate_public_items(r: random.Random) -> list[CorpusItem]:
    projects = [
        "skyline-router", "flexi-grid", "json-flow", "db-sync", "chart-kit",
        "task-runner", "auth-check", "cache-store", "log-fmt", "doc-gen"
    ]
    authors = [
        "dev_coder", "git_guru", "stack_expert", "pr_builder", "code_crafter",
        "sys_admin", "infra_lead", "test_engineer", "api_designer", "web_dev"
    ]
    topics = [
        "performance", "refactoring", "concurrency", "optimization", "asynchronous",
        "serialization", "caching", "validation", "event-driven", "middleware"
    ]

    def make_readme(project, description, install_cmd, code_example):
        return f"""# {project}

{description}

## Installation

Install via package manager:
```bash
{install_cmd}
```

## Quick Start

Import and initialize:
```javascript
{code_example}
```

## Contributing

Please read the contributing guidelines before submitting a pull request.
All contributions must pass lint checks.
"""

    def make_code(project, func_name, param_name, type_name):
        return f"""# {project} helper utility
# Author: open-source contributor

def {func_name}({param_name}: list[{type_name}]) -> dict:
    \"\"\"Process input data and return statistics.\"\"\"
    if not {param_name}:
        return {{"count": 0, "sum": 0.0, "average": 0.0}}
        
    total = sum({param_name})
    avg = total / len({param_name})
    return {{
        "count": len({param_name}),
        "sum": float(total),
        "average": float(avg)
    }}

# Export helper
__all__ = ["{func_name}"]
"""

    def make_issue(project, author, topic, func_name):
        return f"""### Issue: Unhandled exception during {topic} in {project}
Reported by: @{author}

**Description:**
When calling `{func_name}` under high concurrency, the application
sometimes crashes with a TypeError.
This seems to happen when the input array is empty or contains non-numeric values.

**Steps to Reproduce:**
1. Initialize `{project}` client.
2. Call `{func_name}` with an empty list.
3. Observe traceback in console.

**Logs:**
```
TypeError: unsupported operand type(s) for +: 'int' and 'str'
  at {func_name} (utils.py:12)
  at process_request (server.py:45)
```
"""

    def make_changelog(project, version, date, topic):
        return f"""## [{version}] - {date}

### Added
- Integrated new {topic} optimization logic in {project}.
- Added validation for configuration objects.

### Fixed
- Fixed memory leak in {project} buffer allocation.
- Corrected type annotation for constructor parameters.

### Security
- Updated dependency package versions to address upstream vulnerability.
"""

    def make_release(project, version, author, description):
        return f"""# Release {project} v{version}
Released by @{author}

This release introduces major improvements to the core execution pipeline.

### Highlights:
- {description}
- Full backward compatibility with the previous major version.
- Enhanced telemetry hooks for application performance monitoring.

For full list of changes, see the CHANGELOG.md file.
"""

    items = []
    
    # 10 READMEs
    for i in range(10):
        project = projects[i]
        desc = f"A fast and lightweight library for {topics[i]} operations."
        inst = f"npm install @scope/{project}"
        example = (
            f"const {project.replace('-', '_')} = require('{project}');\n"
            f"// Initialize client\n"
            f"const client = new {project.replace('-', '_')}.Client();"
        )
        text = make_readme(project, desc, inst, example)
        items.append(CorpusItem(
            source="sensitivity_corpus",
            text=text,
            label=0,
            label_origin=LabelOrigin.KNOWN,
            category="sensitivity_public",
            sensitivity="public",
            expected_action="route_cloud",
        ))
    
    # 10 Code files
    for i in range(10):
        project = projects[i]
        func = f"process_{topics[i].replace('-', '_')}"
        param = "items"
        tname = "float" if i % 2 == 0 else "int"
        text = make_code(project, func, param, tname)
        items.append(CorpusItem(
            source="sensitivity_corpus",
            text=text,
            label=0,
            label_origin=LabelOrigin.KNOWN,
            category="sensitivity_public",
            sensitivity="public",
            expected_action="route_cloud",
        ))

    # 10 Issues
    for i in range(10):
        project = projects[i]
        author = authors[i]
        topic = topics[(i + 1) % 10]
        func = f"analyze_{topics[i].replace('-', '_')}"
        text = make_issue(project, author, topic, func)
        items.append(CorpusItem(
            source="sensitivity_corpus",
            text=text,
            label=0,
            label_origin=LabelOrigin.KNOWN,
            category="sensitivity_public",
            sensitivity="public",
            expected_action="route_cloud",
        ))

    # 10 Changelogs
    for i in range(10):
        project = projects[i]
        version = f"1.{i}.2"
        date = f"2026-06-{i+1:02d}"
        topic = topics[(i + 2) % 10]
        text = make_changelog(project, version, date, topic)
        items.append(CorpusItem(
            source="sensitivity_corpus",
            text=text,
            label=0,
            label_origin=LabelOrigin.KNOWN,
            category="sensitivity_public",
            sensitivity="public",
            expected_action="route_cloud",
        ))

    # 10 Release notes
    for i in range(10):
        project = projects[i]
        version = f"2.{i}.0"
        author = authors[(i + 3) % 10]
        desc = (
            f"Rewritten engine for {topics[(i + 4) % 10]} "
            f"which improves throughput by {10 + i * 5}%."
        )
        text = make_release(project, version, author, desc)
        items.append(CorpusItem(
            source="sensitivity_corpus",
            text=text,
            label=0,
            label_origin=LabelOrigin.KNOWN,
            category="sensitivity_public",
            sensitivity="public",
            expected_action="route_cloud",
        ))

    return items


def generate_sensitive_doc_items(r: random.Random) -> list[CorpusItem]:
    first_names = [
        "Alice", "Bob", "Charlie", "Diana", "Ethan", "Fiona", "George",
        "Hannah", "Ian", "Julia", "Kevin", "Laura", "Marcus", "Nora",
        "Oscar", "Paula", "Quincy", "Rachel", "Samuel", "Tina", "Victor",
        "Wendy", "Xavier", "Yvonne", "Zachary"
    ]
    last_names = [
        "Anderson", "Brown", "Carter", "Davis", "Evans", "Foster", "Garcia",
        "Harris", "Jackson", "King", "Lewis", "Miller", "Nelson", "Owen",
        "Perez", "Robinson", "Smith", "Taylor", "Thomas", "Walker", "Young"
    ]
    departments = [
        "Engineering", "Product", "Sales", "Marketing", "Finance",
        "HR", "Legal", "Operations", "Support", "Security"
    ]
    titles = [
        "Software Engineer", "Product Manager", "Account Executive",
        "Marketing Specialist", "Financial Analyst", "HR Generalist",
        "Legal Counsel", "Operations Manager", "Support Engineer",
        "Security Analyst"
    ]

    items = []

    # 1. 10 HR Performance Reviews
    for idx in range(10):
        first = r.choice(first_names)
        last = r.choice(last_names)
        name = f"{first} {last}"
        email = f"{first.lower()}.{last.lower()}@acme-corp.invalid"
        ssn = f"{r.randint(100, 999)}-{r.randint(10, 99)}-{r.randint(1000, 9999)}"
        phone = f"({r.randint(200, 999)}) 555-{r.randint(1000, 9999)}"
        dept = departments[idx % len(departments)]
        title = titles[idx % len(titles)]
        salary = r.randint(80, 220) * 1000
        
        text = f"""CONFIDENTIAL EMPLOYEE FILE - DO NOT DISTRIBUTE
Employee Name: {name}
Title: Senior {title}
Department: {dept}
Email: {email}
SSN: {ssn}
Phone: {phone}

Performance Summary:
{name} has consistently met expectations. However, there are minor concerns about collaboration.
Salary for next fiscal year: ${salary:,}/yr.
"""
        items.append(CorpusItem(
            source="sensitivity_corpus",
            text=text,
            label=0,
            label_origin=LabelOrigin.KNOWN,
            category="sensitivity_sensitive_doc",
            sensitivity="sensitive_doc",
            expected_action="route_local",
        ))

    # 2. 10 Invoices
    for _ in range(10):
        first = r.choice(first_names)
        last = r.choice(last_names)
        name = f"{first} {last}"
        email = f"{first.lower()}.{last.lower()}@client-mail.invalid"
        phone = f"({r.randint(200, 999)}) 555-{r.randint(1000, 9999)}"
        card_num = (
            f"4111-{r.randint(1000, 9999)}-{r.randint(1000, 9999)}-{r.randint(1000, 9999)}"
        )
        inv_id = r.randint(10000, 99999)
        address = f"{r.randint(100, 999)} Main St, Cityville, NY {r.randint(10000, 99999)}"
        
        text = f"""ACME CORPORATION - INTERNAL FINANCIAL RECORD
Invoice ID: INV-{inv_id}
Customer Name: {name}
Billing Address: {address}
Contact Email: {email}
Phone: {phone}
Payment Method: Visa ending in {card_num}

Line Items:
1. Enterprise Cloud Subscription - $12,500.00
2. Professional Services Support - $5,000.00
Total Amount Due: $17,500.00
"""
        items.append(CorpusItem(
            source="sensitivity_corpus",
            text=text,
            label=0,
            label_origin=LabelOrigin.KNOWN,
            category="sensitivity_sensitive_doc",
            sensitivity="sensitive_doc",
            expected_action="route_local",
        ))

    # 3. 10 Patient Medical Records
    for _ in range(10):
        first = r.choice(first_names)
        last = r.choice(last_names)
        name = f"{first} {last}"
        email = f"{first.lower()}.{last.lower()}@patient-mail.invalid"
        ssn = f"{r.randint(100, 999)}-{r.randint(10, 99)}-{r.randint(1000, 9999)}"
        phone = f"({r.randint(200, 999)}) 555-{r.randint(1000, 9999)}"
        dob = f"19{r.randint(50, 99)}-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}"
        
        text = f"""VALLEY MEDICAL CENTER - RESTRICTED PATIENT CHART
Patient Name: {name}
Date of Birth: {dob}
Social Security Number: {ssn}
Primary Care Email: {email}
Contact Phone: {phone}

Clinical Notes:
Patient presented with mild symptoms of fatigue. Prescribed vitamin supplements and advised rest.
Follow-up scheduled in 2 weeks.
"""
        items.append(CorpusItem(
            source="sensitivity_corpus",
            text=text,
            label=0,
            label_origin=LabelOrigin.KNOWN,
            category="sensitivity_sensitive_doc",
            sensitivity="sensitive_doc",
            expected_action="route_local",
        ))

    # 4. 10 Support Tickets
    for _ in range(10):
        first = r.choice(first_names)
        last = r.choice(last_names)
        name = f"{first} {last}"
        email = f"{first.lower()}.{last.lower()}@customer.invalid"
        phone = f"({r.randint(200, 999)}) 555-{r.randint(1000, 9999)}"
        card_num = (
            f"5555-{r.randint(1000, 9999)}-{r.randint(1000, 9999)}-{r.randint(1000, 9999)}"
        )
        ticket_id = r.randint(100000, 999999)
        inv_id = r.randint(10000, 99999)
        
        text = f"""SUPPORT TICKET #Dispute-{ticket_id} (RESTRICTED ACCESS)
Opened By: {name}
Customer Email: {email}
Account Phone: {phone}

Description:
I am disputing a charge of $1,200.00 on my credit card ({card_num}).
The invoice number was INV-{inv_id}. I did not authorize this charge. Please investigate.
"""
        items.append(CorpusItem(
            source="sensitivity_corpus",
            text=text,
            label=0,
            label_origin=LabelOrigin.KNOWN,
            category="sensitivity_sensitive_doc",
            sensitivity="sensitive_doc",
            expected_action="route_local",
        ))

    # 5. 10 HR Grievances
    for _ in range(10):
        first = r.choice(first_names)
        last = r.choice(last_names)
        name = f"{first} {last}"
        email = f"{first.lower()}.{last.lower()}@acme-corp.invalid"
        ssn = f"{r.randint(100, 999)}-{r.randint(10, 99)}-{r.randint(1000, 9999)}"
        phone = f"({r.randint(200, 999)}) 555-{r.randint(1000, 9999)}"
        case_id = r.randint(1000, 9999)
        date = f"2026-{r.randint(1, 5):02d}-{r.randint(1, 28):02d}"
        
        text = f"""HUMAN RESOURCES PRIVATE RECORD - PRIVILEGED AND CONFIDENTIAL
Case ID: HRG-{case_id}
Filer Name: {name}
Email: {email}
Contact Number: {phone}
Social Security Number: {ssn}

Statement of Grievance:
Filer reported an issue regarding workspace behavior on {date}.
The HR committee has opened a formal review.
This document is strictly confidential and not for public disclosure.
"""
        items.append(CorpusItem(
            source="sensitivity_corpus",
            text=text,
            label=0,
            label_origin=LabelOrigin.KNOWN,
            category="sensitivity_sensitive_doc",
            sensitivity="sensitive_doc",
            expected_action="route_local",
        ))

    return items


def generate_secret_bearing_items(r: random.Random) -> list[CorpusItem]:
    templates = [
        # Dockerfile
        """FROM node:18-alpine
WORKDIR /app
{SECRET_LINE}
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 3000
CMD ["npm", "start"]""",
        
        # .env
        """# Development environment variables
PORT=8080
DB_HOST=localhost
DB_USER=admin
{SECRET_LINE}
DEBUG=true""",
        
        # GitHub Actions YAML
        """name: Deploy Application
on:
  push:
    branches: [ main ]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run deploy script
        env:
          {SECRET_LINE}
        run: ./scripts/deploy.sh""",
        
        # Kubernetes YAML
        """apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
  namespace: default
type: Opaque
stringData:
  {SECRET_LINE}""",
        
        # Python config
        """#!/usr/bin/env python3
# Automated deployment script for production environments
import os
import sys

# Credentials configuration
{SECRET_LINE}

def run_deploy():
    print("Deploying...")
    # perform remote calls
    print("Deployment finished successfully!")

if __name__ == "__main__":
    run_deploy()""",
        
        # Node.js server
        """// Node.js Server Configuration
const express = require('express');
const app = express();

// Credentials configuration
{SECRET_LINE}

app.get('/api/status', (req, res) => {{
  res.json({{ status: 'online', active: true }});
}});

app.listen(3000, () => console.log('Server running'));""",
        
        # Bash log
        """[CI LOG] [2026-06-09 12:00:00] Starting build step 3: configure keys...
[CI LOG] Running build tool setup...
[CI LOG] Warning: detected hardcoded credential during step execution:
[CI LOG] {SECRET_LINE}
[CI LOG] Build successfully finished. Output saved to artifacts/build.tar.gz."""
    ]

    items = []
    for idx in range(50):
        tmpl_idx = idx % 7
        secret_type = idx % 7
        template = templates[tmpl_idx]
        
        if secret_type == 0:
            secret = f"AKIADUMMYKEY{idx:08d}"
            if tmpl_idx == 0:
                line = f"ENV SECRET_VALUE={secret}"
            elif tmpl_idx == 1:
                line = f"API_CREDENTIAL={secret}"
            elif tmpl_idx == 2:
                line = f"DEPLOY_KEY: {secret}"
            elif tmpl_idx == 3:
                line = f"api_token: {secret}"
            elif tmpl_idx == 4:
                line = f"DEPLOY_TOKEN = '{secret}'"
            elif tmpl_idx == 5:
                line = f"const token = '{secret}';"
            else:
                line = f"FOUND: {secret}"
        elif secret_type == 1:
            secret = f"sk-ProjExampleKeyForTestOnly{idx:03d}XXXXXXXXXXXXXXXXX"
            if tmpl_idx == 0:
                line = f"ENV SECRET_VALUE={secret}"
            elif tmpl_idx == 1:
                line = f"API_CREDENTIAL={secret}"
            elif tmpl_idx == 2:
                line = f"DEPLOY_KEY: {secret}"
            elif tmpl_idx == 3:
                line = f"api_token: {secret}"
            elif tmpl_idx == 4:
                line = f"DEPLOY_TOKEN = '{secret}'"
            elif tmpl_idx == 5:
                line = f"const token = '{secret}';"
            else:
                line = f"FOUND: {secret}"
        elif secret_type == 2:
            secret = f"ghp_ExampleGitHubTokenForTesting{idx:03d}XXXXXX"
            if tmpl_idx == 0:
                line = f"ENV SECRET_VALUE={secret}"
            elif tmpl_idx == 1:
                line = f"API_CREDENTIAL={secret}"
            elif tmpl_idx == 2:
                line = f"DEPLOY_KEY: {secret}"
            elif tmpl_idx == 3:
                line = f"api_token: {secret}"
            elif tmpl_idx == 4:
                line = f"DEPLOY_TOKEN = '{secret}'"
            elif tmpl_idx == 5:
                line = f"const token = '{secret}';"
            else:
                line = f"FOUND: {secret}"
        elif secret_type == 3:
            secret = f"xoxb-ExampleSlackTokenForTesting{idx:03d}XXXXXX"
            if tmpl_idx == 0:
                line = f"ENV SECRET_VALUE={secret}"
            elif tmpl_idx == 1:
                line = f"API_CREDENTIAL={secret}"
            elif tmpl_idx == 2:
                line = f"DEPLOY_KEY: {secret}"
            elif tmpl_idx == 3:
                line = f"api_token: {secret}"
            elif tmpl_idx == 4:
                line = f"DEPLOY_TOKEN = '{secret}'"
            elif tmpl_idx == 5:
                line = f"const token = '{secret}';"
            else:
                line = f"FOUND: {secret}"
        elif secret_type == 4:
            secret = f"AIzaSyExampleGoogleAPIKeyForTest{idx:03d}XXXX"
            if tmpl_idx == 0:
                line = f"ENV SECRET_VALUE={secret}"
            elif tmpl_idx == 1:
                line = f"API_CREDENTIAL={secret}"
            elif tmpl_idx == 2:
                line = f"DEPLOY_KEY: {secret}"
            elif tmpl_idx == 3:
                line = f"api_token: {secret}"
            elif tmpl_idx == 4:
                line = f"DEPLOY_TOKEN = '{secret}'"
            elif tmpl_idx == 5:
                line = f"const token = '{secret}';"
            else:
                line = f"FOUND: {secret}"
        elif secret_type == 5:
            kw = ["password", "api_key", "secret", "token"][idx % 4]
            secret = f'{kw} = "dummy_secret_for_test_{idx:03d}"'
            line = secret
        elif secret_type == 6:
            secret = f"""-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQD{idx:03d}examplekey
-----END PRIVATE KEY-----"""
            line = secret
        else:
            raise ValueError()

        text = template.replace("{SECRET_LINE}", line)
        
        if secret not in text:
            raise ValueError(f"Secret not found in text: {secret}")
        if text.count(secret) != 1:
            raise ValueError(f"Secret occurs {text.count(secret)} times in text: {secret}")
            
        start_idx = text.index(secret)
        end_idx = start_idx + len(secret)
        
        items.append(CorpusItem(
            source="sensitivity_corpus",
            text=text,
            label=0,
            label_origin=LabelOrigin.KNOWN,
            category="sensitivity_secret_bearing",
            sensitivity="secret_bearing",
            expected_action="redact_and_route_cloud",
            planted_secret=secret,
            secret_span=[start_idx, end_idx]
        ))
    return items


def validate_items(items: list[CorpusItem]):
    public_count = 0
    sensitive_doc_count = 0
    secret_bearing_count = 0
    
    for item in items:
        det = detect(item.text)
        cls_res = classify(item.text)
        
        if item.sensitivity == "public":
            public_count += 1
            if len(det) > 0:
                raise ValueError(
                    f"Public item {item.id} triggered detectors: {det}"
                )
            if cls_res.sensitivity != Sensitivity.NONE:
                raise ValueError(
                    f"Public item {item.id} classified as sensitive: "
                    f"{cls_res.sensitivity}"
                )
        
        elif item.sensitivity == "sensitive_doc":
            sensitive_doc_count += 1
            if len(det) == 0:
                raise ValueError(
                    f"Sensitive doc item {item.id} triggered no detectors"
                )
            
            # Should not trigger secret detectors
            detected_types = {d[0] for d in det}
            secret_types = {
                "private_key", "aws_access_key", "openai_key", "github_token",
                "slack_token", "google_api_key", "assignment", "high_entropy_token"
            }
            triggered_secrets = detected_types.intersection(secret_types)
            if triggered_secrets:
                raise ValueError(
                    f"Sensitive doc item {item.id} triggered secret detectors: "
                    f"{triggered_secrets}"
                )
            if cls_res.sensitivity != Sensitivity.PII:
                raise ValueError(
                    f"Sensitive doc item {item.id} classified as "
                    f"{cls_res.sensitivity} instead of PII"
                )
                
        elif item.sensitivity == "secret_bearing":
            secret_bearing_count += 1
            if not item.planted_secret:
                raise ValueError(
                    f"Secret bearing item {item.id} is missing planted_secret field"
                )
            if not item.secret_span:
                raise ValueError(
                    f"Secret bearing item {item.id} is missing secret_span field"
                )
            
            # Verify the secret is actually at the span
            start, end = item.secret_span
            span_text = item.text[start:end]
            if span_text != item.planted_secret:
                raise ValueError(
                    f"Secret span mismatch for {item.id}: "
                    f"text[{start}:{end}] = {span_text!r} vs "
                    f"planted = {item.planted_secret!r}"
                )
                
            # Verify detect finds it
            detected_types = [t for t, _ in det]
            secret_types = {
                "private_key", "aws_access_key", "openai_key", "github_token",
                "slack_token", "google_api_key", "assignment", "high_entropy_token"
            }
            has_secret = any(t in secret_types for t in detected_types)
            if not has_secret:
                raise ValueError(
                    f"Secret bearing item {item.id} triggered no secret detectors: {det}"
                )
            if cls_res.sensitivity != Sensitivity.SECRET:
                raise ValueError(
                    f"Secret bearing item {item.id} classified as "
                    f"{cls_res.sensitivity} instead of SECRET"
                )
                
    print(
        f"Self-check passed: public={public_count}, "
        f"sensitive_doc={sensitive_doc_count}, "
        f"secret_bearing={secret_bearing_count}"
    )


def main():
    # Deterministic generation using seeded randomizers
    r_public = random.Random(42)
    r_sensitive = random.Random(2026)
    r_secret = random.Random(10101)

    print("Generating public items...")
    public_items = generate_public_items(r_public)
    
    print("Generating sensitive_doc items...")
    sensitive_items = generate_sensitive_doc_items(r_sensitive)
    
    print("Generating secret_bearing items...")
    secret_items = generate_secret_bearing_items(r_secret)

    all_items = public_items + sensitive_items + secret_items
    
    print(f"Total items generated: {len(all_items)}")
    print("Validating items...")
    validate_items(all_items)

    print(f"Writing corpus to {SENSITIVITY_CORPUS}...")
    n = write_jsonl(SENSITIVITY_CORPUS, all_items)
    print(f"Successfully wrote {n} items to {SENSITIVITY_CORPUS}")


if __name__ == "__main__":
    main()
