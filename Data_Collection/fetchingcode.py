import requests
import time
import csv
import os
import json
from datetime import datetime
from collections import Counter

TARGET_REPO_COUNT = 10  
CSV_FILE = 'github_data_summarized1.csv'
DATE_RANGES = [
    ('2024-01-01', '2024-06-30'),         
    ('2023-07-01', '2023-12-31'),
    ('2023-01-01', '2023-06-30'),
]

GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
if not GITHUB_TOKEN:
    raise ValueError("github token not found ")

graphql_headers = {"Authorization": f"bearer {GITHUB_TOKEN}"}
rest_headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}

# Updated  fieldnames to reflect the summarized contribution data.
fieldnames = [
    'repository_name', 'owner', 'email', 'stars', 'forks', 'commits', 'total_public_repos',
    'owner_issues_opened', 'owner_issues_closed', 'owner_merged_prs', 'avg_pr_lead_time_days_in_repo',
    'total_contributions_to_others', 'avg_contrib_lines_changed', 'avg_contrib_lead_time_days', 'top_contrib_categories'
]

# --- GraphQL Query (remains the same as it fetches the necessary details) ---
GRAPHQL_QUERY = """
query repoDetails($owner: String!, $repoName: String!, $searchQuery: String!) {
  repository(owner: $owner, name: $repoName) {
    stargazerCount
    forkCount
    mergedPRs: pullRequests(first: 20, states: [MERGED], orderBy: {field: CREATED_AT, direction: DESC}) {
      nodes {
        createdAt
        mergedAt
      }
    }
    defaultBranchRef {
      target {
        ... on Commit {
          history {
            totalCount
          }
        }
      }
    }
  }
  ownerInfo: user(login: $owner) {
    publicRepositories: repositories(first: 1) {
      totalCount
    }
    openedIssues: issues {
      totalCount
    }
    closedIssues: issues(states: CLOSED) {
      totalCount
    }
    mergedPRs: pullRequests(states: MERGED) {
      totalCount
    }
  }
  contributions: search(query: $searchQuery, type: ISSUE, first: 100) {
    nodes {
      ... on PullRequest {
        title
        createdAt
        mergedAt
        additions
        deletions
        repository { nameWithOwner }
        labels(first: 5) { nodes { name } }
      }
    }
  }
}
"""

def parse_datetime(date_string):
    if date_string:
        return datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%SZ')
    return None

def infer_pr_category(title, labels):
    title_lower = title.lower()
    label_names = {label.lower() for label in labels}
    if any(kw in title_lower or f"bug" in label_names for kw in ['fix', 'bug', 'patch']):
        return "Bugfix"
    if any(kw in title_lower or f"feature" in label_names for kw in ['feat', 'feature', 'add']):
        return "Feature"
    if any(kw in title_lower or f"refactor" in label_names for kw in ['refactor', 'perf', 'performance']):
        return "Refactor"
    if any(kw in title_lower or f"docs" in label_names for kw in ['doc', 'docs', 'documentation']):
        return "Documentation"
    return "Enhancement"

def get_latest_commit_email(owner, repo_name):
    try:
        url = f"https://api.github.com/repos/{owner}/{repo_name}/commits"
        response = requests.get(url, headers=rest_headers, params={'per_page': 1})
        response.raise_for_status()
        commits = response.json()
        if commits:
            return commits[0]['commit']['author'].get('email', 'N/A')
    except requests.exceptions.RequestException:
        return 'N/A'

# --- MAIN EXECUTION ---
with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    total_repos_saved = 0
    processed_repos = set()

    for start_date, end_date in DATE_RANGES:
        if total_repos_saved >= TARGET_REPO_COUNT:
            break
        for page in range(1, 11):
            if total_repos_saved >= TARGET_REPO_COUNT:
                break

            print(f"Searching page {page} for repos created between {start_date} and {end_date}...")
            params = {'q': f'stars:>20 created:{start_date}..{end_date}', 'sort': 'stars', 'order': 'desc', 'per_page': 100, 'page': page}
            try:
                response = requests.get('https://api.github.com/search/repositories', headers=rest_headers, params=params)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                print(f"Error searching for repositories: {e}. Sleeping for 60s.")
                time.sleep(60)
                continue

            repositories = response.json().get('items', [])
            if not repositories:
                break

            for repo in repositories:
                if total_repos_saved >= TARGET_REPO_COUNT:
                    break

                full_name = repo['full_name']
                if full_name in processed_repos or repo['owner']['type'] != 'User':
                    continue

                owner, repository_name = full_name.split('/')
                print(f"({total_repos_saved + 1}/{TARGET_REPO_COUNT}) Processing {full_name}...")

                try:
                    search_query_string = f"is:pr is:merged author:{owner}"
                    variables = {"owner": owner, "repoName": repository_name, "searchQuery": search_query_string}
                    response = requests.post("https://api.github.com/graphql", json={'query': GRAPHQL_QUERY, 'variables': variables}, headers=graphql_headers)
                    response.raise_for_status()
                    data = response.json().get('data', {})

                    if not data or not data.get('repository'):
                        print(f"  - No data returned for {full_name}. Skipping.")
                        continue

                    repo_data = data.get('repository', {})
                    owner_data = data.get('ownerInfo', {})
                    contributions_data = data.get('contributions', {})

                    lead_times = []
                    for pr in repo_data.get('mergedPRs', {}).get('nodes', []):
                        created_at = parse_datetime(pr.get('createdAt'))
                        merged_at = parse_datetime(pr.get('mergedAt'))
                        if created_at and merged_at:
                            lead_times.append((merged_at - created_at).total_seconds())
                    avg_lead_time_days = (sum(lead_times) / len(lead_times) / 86400) if lead_times else 0

                    email = get_latest_commit_email(owner, repository_name)
                    commits_history = repo_data.get('defaultBranchRef', {}).get('target', {}).get('history', {}) or {}

                    # SUMMARIZE CONTRIBUTION DATA ---
                    contrib_lines = []
                    contrib_lead_days = []
                    contrib_categories = []
                    
                    for pr in contributions_data.get('nodes', []):
                        repo_with_owner = pr.get('repository', {}).get('nameWithOwner')
                        if repo_with_owner and not repo_with_owner.startswith(f"{owner}/"):
                            contrib_lines.append(pr.get('additions', 0) + pr.get('deletions', 0))
                            
                            created = parse_datetime(pr.get('createdAt'))
                            merged = parse_datetime(pr.get('mergedAt'))
                            if created and merged:
                                contrib_lead_days.append((merged - created).days)
                            
                            contrib_categories.append(infer_pr_category(pr.get('title', ''), [l.get('name') for l in pr.get('labels', {}).get('nodes', [])]))

                    # Calculate summary statistics
                    total_contributions_to_others = len(contrib_lines)
                    avg_contrib_lines_changed = sum(contrib_lines) / total_contributions_to_others if total_contributions_to_others > 0 else 0
                    avg_contrib_lead_time_days = sum(contrib_lead_days) / len(contrib_lead_days) if contrib_lead_days else 0
                    
                    category_counts = Counter(contrib_categories)
                    top_contrib_categories = ", ".join([cat for cat, count in category_counts.most_common(2)])

                    #  SUMMARIZED ROW 
                    writer.writerow({
                        'repository_name': repository_name,
                        'owner': owner,
                        'email': email,
                        'stars': repo_data.get('stargazerCount', 0),
                        'forks': repo_data.get('forkCount', 0),
                        'commits': commits_history.get('totalCount', 0),
                        'total_public_repos': owner_data.get('publicRepositories', {}).get('totalCount', 0),
                        'owner_issues_opened': owner_data.get('openedIssues', {}).get('totalCount', 'N/A'),
                        'owner_issues_closed': owner_data.get('closedIssues', {}).get('totalCount', 'N/A'),
                        'owner_merged_prs': owner_data.get('mergedPRs', {}).get('totalCount', 'N/A'),
                        'avg_pr_lead_time_days_in_repo': round(avg_lead_time_days, 2),
                        'total_contributions_to_others': total_contributions_to_others,
                        'avg_contrib_lines_changed': round(avg_contrib_lines_changed, 2),
                        'avg_contrib_lead_time_days': round(avg_contrib_lead_time_days, 2),
                        'top_contrib_categories': top_contrib_categories
                    })

                    processed_repos.add(full_name)
                    total_repos_saved += 1

                except requests.exceptions.RequestException as e:
                    print(f"  - Error processing {full_name}: {e}")
                except Exception as e:
                    print(f"  - An unexpected error occurred while processing {full_name}: {e}")

                time.sleep(0.5)

print(f"\n  Data collection complete. Total unique repositories processed: {total_repos_saved}")