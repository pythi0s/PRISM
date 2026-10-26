#Automated Review checker

This Project check alignment between Pull Requests & JIRA ticket to check whether
the changes are aligned or Not.

It further PRs as High Risk or Low Risk PRs.

To Run the step up perform the following steps:
```bash
pip3 install -r requirement.txt
python3 data_generation.py
python3 check_pr_jira_alignment.py
python3 detect_high_risk_pr.py