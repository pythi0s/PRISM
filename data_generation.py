import pandas as pd
from datetime import datetime, timedelta
import random

# Constants
NUM_RECORDS = 500
START_DATE = datetime(2025, 3, 1)
END_DATE = datetime(2025, 5, 15)
REPO = "my-python-app"
USERS = [
    "john.doe@example.com", "jane.smith@example.com", "alex.brown@example.com",
    "sarah.lee@example.com", "mike.wilson@example.com", "emma.jones@example.com",
    "liam.martin@example.com", "olivia.taylor@example.com", "sophia.white@example.com",
    "james.moore@example.com"
]
TICKET_TYPES = ["Feature", "Bug", "Improvement"]
PRIORITIES = ["Low", "Medium", "High"]
STATUSES = ["Open", "In Progress", "Closed"]

# Task templates
TASKS = [
    {
        "type": "Feature",
        "title": "Add {item} endpoint",
        "description": "Implement {item} API endpoint using {tech}",
        "pr_title": "Implement {item} endpoint",
        "pr_description": "Added {item} API endpoint with {tech}",
        "branch_prefix": "feature",
        "code_summary": "Added {module}.py with {tech} implementation",
        "items": ["user authentication", "email notifications", "Redis caching", "password reset", "search pagination"],
        "techs": ["JWT", "smtplib", "Redis", "OAuth", "Elasticsearch"]
    },
    {
        "type": "Bug",
        "title": "Fix {item} bug",
        "description": "Resolve {item} issue in {module} module",
        "pr_title": "Fix {item} in {module}",
        "pr_description": "Fixed {item} in {module} module",
        "branch_prefix": "bugfix",
        "code_summary": "Updated {module}.py to fix {item}",
        "items": ["database timeouts", "null API responses", "CSV parsing", "session expiry", "SQL injection"],
        "techs": ["SQLAlchemy", "pandas", "Flask", "Django", "PyMySQL"]
    },
    {
        "type": "Improvement",
        "title": "Optimize {item}",
        "description": "Improve {item} performance using {tech}",
        "pr_title": "Optimize {item} performance",
        "pr_description": "Enhanced {item} with {tech} for better performance",
        "branch_prefix": "improvement",
        "code_summary": "Refactored {module}.py with {tech}",
        "items": ["data pipeline", "logging", "API performance", "file uploads", "query execution"],
        "techs": ["multiprocessing", "JSONFormatter", "Memcached", "asyncio", "NumPy"]
    }
]
MODULES = ["auth", "database", "pipeline", "email", "user", "search", "logger", "api", "file"]

FILE_PATHS = [
    "src/auth/auth.py", "src/auth/utils.py", "src/database/models.py", "src/database/queries.py",
    "src/pipeline/processor.py", "src/email/sender.py", "src/user/models.py", "src/search/index.py",
    "src/logger/logger.py", "src/api/views.py", "src/file/uploader.py", "tests/test_auth.py",
    "tests/test_database.py", "tests/test_pipeline.py", "tests/test_email.py", "tests/test_user.py",
    "tests/test_search.py", "tests/test_logger.py", "tests/test_api.py", "tests/test_file.py"
]

critical_modules = ['auth', 'database', 'security']
file_risks = {
    "src/auth/auth.py": 0.9,
    "src/database/models.py": 0.8,
    "src/email/sender.py": 0.7,
    "tests/test_auth.py": 0.5,
    "src/logger/logger.py": 0.4,
    "default": 0.3
}

def avg_file_risk(files):
    return sum(file_risks.get(f, file_risks['default']) for f in files) / len(files)

def generate_jira_and_pr():
    jira_data, pr_data = [], []

    for i in range(1001, 1501):
        ticket_id = f"PROJ-{i}"
        task = random.choice(TASKS)
        item = random.choice(task["items"])
        tech = random.choice(task["techs"])
        module = random.choice(MODULES) if task["type"] == "Bug" else item.replace(" ", "_")
        title = task["title"].format(item=item)
        description = task["description"].format(item=item, tech=tech, module=module)
        status = random.choice(STATUSES)
        priority = random.choice(PRIORITIES)
        assignee = random.choice(USERS)
        created_date = START_DATE + timedelta(days=random.randint(0, 60))
        updated_date = created_date + timedelta(days=random.randint(1, (END_DATE - created_date).days))

        jira_data.append({
            "id": ticket_id, "title": title, "description": description,
            "type": task["type"], "status": status, "priority": priority,
            "assignee": assignee, "created_date": created_date.strftime("%Y-%m-%d"),
            "updated_date": updated_date.strftime("%Y-%m-%d"), "_item": item,
            "_tech": tech, "_module": module
        })

        pr_id = f"PR-{i}"
        pr_title = task["pr_title"].format(item=item, module=module)
        pr_description = task["pr_description"].format(item=item, tech=tech, module=module)
        branch = f"{task['branch_prefix']}/{ticket_id}-{item.replace(' ', '-')}"
        pr_status = "Merged" if status == "Closed" else "Open"
        pr_created = created_date + timedelta(days=random.randint(1, 10))
        code_summary = task["code_summary"].format(module=module, tech=tech, item=item)
        files_changed = random.sample(FILE_PATHS, random.randint(1, 5))
        lines_added = random.randint(10, 200)
        lines_deleted = random.randint(5, 100)
        comment_count = random.randint(0, 20)

        pr_data.append({
            "pr_id": pr_id, "jira_id": ticket_id, "title": pr_title, "description": pr_description,
            "repo": REPO, "branch": branch, "status": pr_status, "created_date": pr_created.strftime("%Y-%m-%d"),
            "author": assignee, "code_change_summary": code_summary, "files_changed": files_changed,
            "lines_added": lines_added, "lines_deleted": lines_deleted, "comment_count": comment_count,
            "title_length": len(pr_title), "description_length": len(pr_description),
            "has_numbers_in_title": int(any(char.isdigit() for char in pr_title)),
            "avg_file_risk": avg_file_risk(files_changed),
            "has_critical_module": int(any(mod in f for f in files_changed for mod in critical_modules)),
            "code_churn_ratio": lines_added / (lines_deleted + 1),
            "semantic_mismatch": int(pr_title.split()[0].lower() != pr_description.split()[0].lower())
        })

    return jira_data, pr_data

# Generate data
jira_data, pr_data = generate_jira_and_pr()

# Inject 10% misaligned PRs
num_misaligned = int(0.10 * len(pr_data))
misaligned_indices = random.sample(range(len(pr_data)), num_misaligned)
for idx in misaligned_indices:
    if random.random() < 0.33:
        pr_data[idx]['title'] = "Update documentation and minor refactor"
    elif random.random() < 0.66:
        pr_data[idx]['description'] = "Minor updates to README and cleanup in unrelated modules"
    else:
        other_jira = random.choice([j for j in jira_data if j['id'] != pr_data[idx]['jira_id']])
        pr_data[idx]['jira_id'] = other_jira['id']

# Save to CSV
jira_df = pd.DataFrame([{k: v for k, v in d.items() if not k.startswith('_')} for d in jira_data])
pr_df = pd.DataFrame(pr_data)

# Additional time-based features
pr_df['created_date'] = pd.to_datetime(pr_df['created_date'])
pr_df['day_of_week'] = pr_df['created_date'].dt.dayofweek
pr_df = pr_df.sort_values(by=['author', 'created_date'])
pr_df['time_since_last_pr'] = pr_df.groupby('author')['created_date'].diff().fillna(pd.Timedelta(days=0)).dt.days

jira_df.to_csv("jira_tickets.csv", index=False)
pr_df.to_csv("pull_requests.csv", index=False)

print("✅ Enhanced synthetic data generated and saved.")