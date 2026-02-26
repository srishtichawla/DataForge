from typing import List, Optional
import json
import csv
import random
import string
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

LOREM_WORDS = (
    "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "
    "incididunt ut labore et dolore magna aliqua enim ad minim veniam quis nostrud "
    "exercitation ullamco laboris nisi aliquip ex ea commodo consequat duis aute irure "
    "dolor reprehenderit voluptate velit esse cillum dolore eu fugiat nulla pariatur "
    "excepteur sint occaecat cupidatat non proident sunt culpa qui officia deserunt "
    "mollit anim id est laborum"
).split()

STREET_SUFFIXES = ["St", "Ave", "Blvd", "Rd", "Ln", "Dr", "Ct", "Pl", "Way", "Terrace"]
CITIES = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia",
    "San Antonio", "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville",
    "London", "Berlin", "Paris", "Tokyo", "Sydney", "Toronto", "Vancouver",
]
COUNTRIES = ["USA", "UK", "Germany", "France", "Japan", "Australia", "Canada"]
STATES = ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]
JOB_TITLES = [
    "Software Engineer", "Product Manager", "Data Scientist", "UX Designer",
    "DevOps Engineer", "Marketing Manager", "Sales Representative", "Analyst",
    "HR Manager", "Operations Lead", "QA Engineer", "Technical Writer",
    "CTO", "CFO", "CEO", "COO", "VP Engineering", "Director of Product",
]
DEPARTMENTS = ["Engineering", "Product", "Marketing", "Sales", "HR", "Finance", "Operations", "Design"]
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CAD", "AUD"]
PRODUCT_ADJECTIVES = ["Premium", "Classic", "Ultra", "Smart", "Pro", "Lite", "Advanced", "Essential"]
PRODUCT_NOUNS = ["Widget", "Gadget", "Device", "Module", "Kit", "Pack", "Bundle", "Set"]
CATEGORIES = ["Electronics", "Clothing", "Home & Garden", "Sports", "Books", "Food", "Toys", "Automotive"]


def _rand_phone() -> str:
    return f"+1-{random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}"


def _rand_address() -> dict:
    return {
        "street": f"{random.randint(1, 9999)} {random.choice(string.ascii_uppercase + string.ascii_uppercase)}"
                  f"{''.join(random.choices(string.ascii_lowercase, k=6))} {random.choice(STREET_SUFFIXES)}",
        "city": random.choice(CITIES),
        "state": random.choice(STATES),
        "zip": f"{random.randint(10000, 99999)}",
        "country": random.choice(COUNTRIES),
    }


def _rand_date(start_days_ago: int = 730, end_days_ago: int = 0) -> str:
    delta = random.randint(end_days_ago, start_days_ago)
    return (datetime.now() - timedelta(days=delta)).strftime("%Y-%m-%dT%H:%M:%S")


def _lorem(words: int) -> str:
    chosen = [random.choice(LOREM_WORDS) for _ in range(words)]
    chosen[0] = chosen[0].capitalize()
    return " ".join(chosen) + "."


# ─────────────────────────────────────────────
#  File I/O Tools
# ─────────────────────────────────────────────

@tool
def write_json(filepath: str, data: dict) -> str:
    """Write a Python dictionary as formatted JSON to a file."""
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        size = os.path.getsize(filepath)
        return f"✓ Wrote JSON to '{filepath}' ({size:,} bytes, {len(json.dumps(data))} chars)."
    except Exception as e:
        return f"Error writing JSON: {e}"


@tool
def read_json(filepath: str) -> str:
    """Read and return the contents of a JSON file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data, indent=2)
    except FileNotFoundError:
        return f"Error: '{filepath}' not found."
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON — {e}"
    except Exception as e:
        return f"Error: {e}"


@tool
def write_csv(filepath: str, records: list) -> str:
    """
    Write a list of flat dictionaries as a CSV file.
    All records should have the same keys (used as column headers).
    """
    if not records:
        return "Error: records list is empty."
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        headers = list(records[0].keys())
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(records)
        size = os.path.getsize(filepath)
        return f"✓ Wrote {len(records)} rows to CSV '{filepath}' ({size:,} bytes)."
    except Exception as e:
        return f"Error writing CSV: {e}"


@tool
def read_csv(filepath: str) -> str:
    """Read a CSV file and return its contents as a JSON string."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return json.dumps(rows, indent=2)
    except FileNotFoundError:
        return f"Error: '{filepath}' not found."
    except Exception as e:
        return f"Error: {e}"


@tool
def list_output_files(directory: str = ".") -> str:
    """List all JSON and CSV files in the specified directory."""
    try:
        p = Path(directory)
        if not p.exists():
            return f"Directory '{directory}' does not exist."
        files = sorted(p.glob("**/*.json")) + sorted(p.glob("**/*.csv"))
        if not files:
            return f"No JSON or CSV files found in '{directory}'."
        lines = []
        for fp in files:
            stat = fp.stat()
            lines.append(f"  {fp}  ({stat.st_size:,} bytes, modified {datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')})")
        return "Files found:\n" + "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


# ─────────────────────────────────────────────
#  User Generator
# ─────────────────────────────────────────────

@tool
def generate_users(
    count: int,
    first_names: Optional[List[str]] = None,
    last_names: Optional[List[str]] = None,
    domains: Optional[List[str]] = None,
    min_age: int = 18,
    max_age: int = 65,
    include_address: bool = False,
    include_phone: bool = False,
    include_job: bool = False,
) -> dict:
    """
    Generate realistic sample user records.

    Args:
        count: Number of users to generate (1–500).
        first_names: Optional list of first names to cycle through. Defaults to built-in list.
        last_names: Optional list of last names to cycle through. Defaults to built-in list.
        domains: Email domains to cycle through (e.g. ['example.com']). Defaults to ['example.com'].
        min_age: Minimum age (default 18).
        max_age: Maximum age (default 65).
        include_address: Whether to add a nested address object.
        include_phone: Whether to add a phone number field.
        include_job: Whether to add jobTitle and department fields.

    Returns:
        dict with 'users' list and 'count'.
    """
    DEFAULT_FIRSTS = [
        "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
        "Isabella", "Jack", "Kate", "Liam", "Mia", "Noah", "Olivia", "Peter",
        "Quinn", "Rachel", "Sam", "Taylor", "Uma", "Victor", "Wendy", "Xander",
        "Yasmine", "Zoe",
    ]
    DEFAULT_LASTS = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
        "Davis", "Martinez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore",
        "Jackson", "Martin", "Lee", "Perez", "Thompson", "White",
    ]

    if count < 1 or count > 500:
        return {"error": "count must be between 1 and 500."}
    if min_age > max_age:
        return {"error": f"min_age ({min_age}) > max_age ({max_age})."}

    firsts = first_names or DEFAULT_FIRSTS
    lasts = last_names or DEFAULT_LASTS
    doms = domains or ["example.com"]

    users = []
    for i in range(count):
        first = firsts[i % len(firsts)]
        last = lasts[i % len(lasts)]
        domain = doms[i % len(doms)]
        uid = str(i + 1).zfill(4)

        user: dict = {
            "id": i + 1,
            "uuid": f"usr-{''.join(random.choices(string.hexdigits[:16], k=8))}-{uid}",
            "firstName": first,
            "lastName": last,
            "email": f"{first.lower()}.{last.lower()}@{domain}",
            "username": f"{first.lower()}{random.randint(10, 9999)}",
            "age": random.randint(min_age, max_age),
            "isActive": random.choice([True, True, True, False]),
            "registeredAt": _rand_date(730),
            "lastLoginAt": _rand_date(30),
        }
        if include_phone:
            user["phone"] = _rand_phone()
        if include_address:
            user["address"] = _rand_address()
        if include_job:
            user["jobTitle"] = random.choice(JOB_TITLES)
            user["department"] = random.choice(DEPARTMENTS)
        users.append(user)

    return {"users": users, "count": len(users)}


# ─────────────────────────────────────────────
#  Product Generator
# ─────────────────────────────────────────────

@tool
def generate_products(
    count: int,
    categories: Optional[List[str]] = None,
    min_price: float = 5.0,
    max_price: float = 999.99,
    include_inventory: bool = True,
    include_description: bool = True,
) -> dict:
    """
    Generate realistic sample product records for an e-commerce catalogue.

    Args:
        count: Number of products to generate (1–200).
        categories: Product category names. Defaults to built-in list.
        min_price: Minimum product price (USD).
        max_price: Maximum product price (USD).
        include_inventory: Add stock / inventory fields.
        include_description: Add a lorem-ipsum description.

    Returns:
        dict with 'products' list and 'count'.
    """
    if count < 1 or count > 200:
        return {"error": "count must be between 1 and 200."}
    if min_price > max_price:
        return {"error": f"min_price ({min_price}) > max_price ({max_price})."}

    cats = categories or CATEGORIES
    products = []

    for i in range(count):
        adj = random.choice(PRODUCT_ADJECTIVES)
        noun = random.choice(PRODUCT_NOUNS)
        name = f"{adj} {noun} {random.randint(100, 9999)}"
        price = round(random.uniform(min_price, max_price), 2)

        product: dict = {
            "id": i + 1,
            "sku": f"SKU-{''.join(random.choices(string.ascii_uppercase, k=3))}-{random.randint(1000,9999)}",
            "name": name,
            "category": random.choice(cats),
            "price": price,
            "currency": "USD",
            "rating": round(random.uniform(1.0, 5.0), 1),
            "reviewCount": random.randint(0, 5000),
            "createdAt": _rand_date(365),
        }
        if include_inventory:
            product["stock"] = random.randint(0, 500)
            product["inStock"] = product["stock"] > 0
            product["warehouse"] = random.choice(["US-EAST", "US-WEST", "EU-CENTRAL", "APAC"])
        if include_description:
            product["description"] = _lorem(random.randint(12, 25))
        products.append(product)

    return {"products": products, "count": len(products)}


# ─────────────────────────────────────────────
#  Transaction Generator
# ─────────────────────────────────────────────

@tool
def generate_transactions(
    count: int,
    user_ids: Optional[List[int]] = None,
    product_ids: Optional[List[int]] = None,
    min_amount: float = 1.0,
    max_amount: float = 500.0,
    statuses: Optional[List[str]] = None,
) -> dict:
    """
    Generate realistic financial transaction / order records.

    Args:
        count: Number of transactions (1–1000).
        user_ids: Pool of user IDs to reference. Defaults to [1..20].
        product_ids: Pool of product IDs to reference. Defaults to [1..50].
        min_amount: Minimum transaction amount.
        max_amount: Maximum transaction amount.
        statuses: List of possible statuses. Defaults to ['completed','pending','failed','refunded'].

    Returns:
        dict with 'transactions' list and 'count'.
    """
    if count < 1 or count > 1000:
        return {"error": "count must be between 1 and 1000."}

    uids = user_ids or list(range(1, 21))
    pids = product_ids or list(range(1, 51))
    sts = statuses or ["completed", "completed", "completed", "pending", "failed", "refunded"]

    transactions = []
    for i in range(count):
        amount = round(random.uniform(min_amount, max_amount), 2)
        tax = round(amount * 0.08, 2)
        txn: dict = {
            "id": i + 1,
            "transactionId": f"TXN-{''.join(random.choices(string.ascii_uppercase + string.digits, k=12))}",
            "userId": random.choice(uids),
            "productId": random.choice(pids),
            "quantity": random.randint(1, 5),
            "amount": amount,
            "tax": tax,
            "total": round(amount + tax, 2),
            "currency": random.choice(CURRENCIES),
            "status": random.choice(sts),
            "paymentMethod": random.choice(["credit_card", "debit_card", "paypal", "bank_transfer", "crypto"]),
            "createdAt": _rand_date(365),
        }
        transactions.append(txn)

    return {"transactions": transactions, "count": len(transactions)}


# ─────────────────────────────────────────────
#  Post / Blog Generator
# ─────────────────────────────────────────────

@tool
def generate_posts(
    count: int,
    author_ids: Optional[List[int]] = None,
    include_comments: bool = False,
    max_comments_per_post: int = 5,
    tags: Optional[List[str]] = None,
) -> dict:
    """
    Generate sample blog-post / article records with optional nested comments.

    Args:
        count: Number of posts (1–100).
        author_ids: Pool of author user IDs. Defaults to [1..10].
        include_comments: Embed comment objects inside each post.
        max_comments_per_post: Max number of nested comments (if include_comments=True).
        tags: Tag pool. Defaults to tech/lifestyle list.

    Returns:
        dict with 'posts' list and 'count'.
    """
    if count < 1 or count > 100:
        return {"error": "count must be between 1 and 100."}

    default_tags = ["technology", "science", "health", "lifestyle", "travel", "food",
                    "finance", "education", "sports", "entertainment", "programming", "design"]
    tag_pool = tags or default_tags
    auts = author_ids or list(range(1, 11))

    posts = []
    for i in range(count):
        title_words = [w.capitalize() for w in random.choices(LOREM_WORDS, k=random.randint(4, 9))]
        post: dict = {
            "id": i + 1,
            "slug": "-".join(title_words).lower()[:60],
            "title": " ".join(title_words),
            "body": " ".join([_lorem(random.randint(20, 40)) for _ in range(random.randint(2, 4))]),
            "authorId": random.choice(auts),
            "tags": random.sample(tag_pool, k=random.randint(1, min(4, len(tag_pool)))),
            "published": random.choice([True, True, False]),
            "views": random.randint(0, 50000),
            "likes": random.randint(0, 2000),
            "createdAt": _rand_date(730),
            "updatedAt": _rand_date(30),
        }
        if include_comments:
            num_comments = random.randint(0, max_comments_per_post)
            post["comments"] = [
                {
                    "id": j + 1,
                    "postId": i + 1,
                    "authorId": random.choice(auts),
                    "body": _lorem(random.randint(8, 20)),
                    "likes": random.randint(0, 100),
                    "createdAt": _rand_date(30),
                }
                for j in range(num_comments)
            ]
        posts.append(post)

    return {"posts": posts, "count": len(posts)}


# ─────────────────────────────────────────────
#  Company Generator
# ─────────────────────────────────────────────

INDUSTRIES = [
    "Technology", "Healthcare", "Finance", "Retail", "Manufacturing",
    "Education", "Real Estate", "Transportation", "Media", "Energy",
    "Hospitality", "Consulting", "Legal", "Agriculture", "Aerospace",
]
COMPANY_SUFFIXES = ["Inc.", "LLC", "Ltd.", "Corp.", "Group", "Solutions", "Technologies", "Partners", "Ventures", "Co."]
COMPANY_PREFIXES = [
    "Apex", "Blue", "Core", "Delta", "Echo", "Fusion", "Global", "Horizon",
    "Iris", "Jade", "Kilo", "Luna", "Metro", "Nova", "Orbit", "Peak",
    "Quantum", "Rapid", "Silver", "Titan", "Ultra", "Vertex", "Wave", "Zenith",
]


@tool
def generate_companies(
    count: int,
    industries: Optional[List[str]] = None,
    min_employees: int = 5,
    max_employees: int = 10000,
    include_financials: bool = True,
    include_contact: bool = True,
) -> dict:
    """
    Generate realistic company / organisation records.

    Args:
        count: Number of companies to generate (1-200).
        industries: Industry types to sample from. Defaults to built-in list.
        min_employees: Minimum employee count.
        max_employees: Maximum employee count.
        include_financials: Add annualRevenue, fundingStage, stockTicker fields.
        include_contact: Add website, phone, headquarters address fields.

    Returns:
        dict with 'companies' list and 'count'.
    """
    if count < 1 or count > 200:
        return {"error": "count must be between 1 and 200."}

    ind_pool = industries or INDUSTRIES
    funding_stages = ["Bootstrapped", "Pre-Seed", "Seed", "Series A", "Series B", "Series C", "Public"]
    companies = []

    for i in range(count):
        prefix = random.choice(COMPANY_PREFIXES)
        suffix = random.choice(COMPANY_SUFFIXES)
        name = f"{prefix} {suffix}"
        employees = random.randint(min_employees, max_employees)
        industry = random.choice(ind_pool)

        company: dict = {
            "id": i + 1,
            "name": name,
            "industry": industry,
            "founded": random.randint(1950, 2023),
            "employees": employees,
            "size": (
                "Startup" if employees < 50 else
                "Small" if employees < 250 else
                "Medium" if employees < 1000 else
                "Enterprise"
            ),
            "description": _lorem(random.randint(10, 20)),
            "createdAt": _rand_date(365),
        }
        if include_financials:
            revenue_base = employees * random.randint(50000, 500000)
            company["annualRevenueMillion"] = round(revenue_base / 1_000_000, 2)
            company["revenueCurrency"] = "USD"
            company["fundingStage"] = random.choice(funding_stages)
            if random.random() > 0.6:
                company["stockTicker"] = "".join(random.choices(string.ascii_uppercase, k=random.randint(3, 4)))
        if include_contact:
            company["website"] = f"https://www.{prefix.lower()}.com"
            company["phone"] = _rand_phone()
            company["headquarters"] = _rand_address()

        companies.append(company)

    return {"companies": companies, "count": len(companies)}


# ─────────────────────────────────────────────
#  Event Generator
# ─────────────────────────────────────────────

EVENT_TYPES = ["Conference", "Webinar", "Workshop", "Meetup", "Hackathon", "Summit", "Training", "Networking", "Launch", "AMA"]
EVENT_TOPICS = ["AI & Machine Learning", "Web Development", "Cybersecurity", "Data Science", "Design",
                "Product", "Marketing", "Finance", "Leadership", "DevOps"]
VENUES = ["Grand Ballroom", "Tech Hub", "Innovation Centre", "Community Hall",
          "Rooftop Terrace", "Main Auditorium", "Conference Room A", "Online - Zoom", "Online - Teams"]
SPEAKER_NAMES = [
    "Dr. Sarah Chen", "James Okafor", "Priya Nair", "Marcus Webb",
    "Lena Muller", "Kenji Tanaka", "Fatima Al-Hassan", "Tom Eriksson",
    "Rachel Kim", "David Osei", "Ananya Sharma", "Carlos Rivera",
]


@tool
def generate_events(
    count: int,
    event_types: Optional[List[str]] = None,
    min_attendees: int = 10,
    max_attendees: int = 500,
    include_speakers: bool = True,
    include_tickets: bool = True,
    future_only: bool = False,
) -> dict:
    """
    Generate realistic calendar event / conference records.

    Args:
        count: Number of events to generate (1-200).
        event_types: Types of events to sample from. Defaults to built-in list.
        min_attendees: Minimum attendee count.
        max_attendees: Maximum attendee count.
        include_speakers: Add a list of speaker names and bios.
        include_tickets: Add ticketPrice and ticketUrl fields.
        future_only: If True, all events are in the future.

    Returns:
        dict with 'events' list and 'count'.
    """
    if count < 1 or count > 200:
        return {"error": "count must be between 1 and 200."}

    et_pool = event_types or EVENT_TYPES
    events = []

    for i in range(count):
        topic = random.choice(EVENT_TOPICS)
        etype = random.choice(et_pool)
        title = f"{etype}: {topic} {random.randint(2024, 2026)}"

        if future_only:
            start_dt = datetime.now() + timedelta(days=random.randint(1, 180))
        else:
            start_dt = datetime.now() + timedelta(days=random.randint(-365, 180))

        duration_h = random.choice([1, 2, 3, 4, 6, 8])
        end_dt = start_dt + timedelta(hours=duration_h)

        event: dict = {
            "id": i + 1,
            "title": title,
            "type": etype,
            "topic": topic,
            "status": "upcoming" if start_dt > datetime.now() else "completed",
            "startAt": start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "endAt": end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "durationHours": duration_h,
            "venue": random.choice(VENUES),
            "city": random.choice(CITIES),
            "attendees": random.randint(min_attendees, max_attendees),
            "maxCapacity": max_attendees + random.randint(0, 100),
            "tags": random.sample(EVENT_TOPICS, k=random.randint(1, 3)),
        }
        if include_speakers:
            event["speakers"] = [
                {
                    "name": random.choice(SPEAKER_NAMES),
                    "topic": _lorem(random.randint(4, 7)).rstrip("."),
                    "bio": _lorem(random.randint(8, 14)),
                }
                for _ in range(random.randint(1, 4))
            ]
        if include_tickets:
            event["ticketPrice"] = random.choice([0, 0, 29, 49, 99, 149, 299, 499])
            event["isFree"] = event["ticketPrice"] == 0
            event["ticketUrl"] = f"https://tickets.example.com/event/{i + 1}"

        events.append(event)

    return {"events": events, "count": len(events)}


# ─────────────────────────────────────────────
#  Invoice Generator
# ─────────────────────────────────────────────

SERVICE_ITEMS = [
    "Consulting Services", "Software Development", "Design Work", "Data Analysis",
    "Marketing Campaign", "SEO Audit", "Cloud Infrastructure", "Support & Maintenance",
    "Training Session", "Project Management", "Content Writing", "Legal Review",
]


@tool
def generate_invoices(
    count: int,
    client_ids: Optional[List[int]] = None,
    min_line_items: int = 1,
    max_line_items: int = 6,
    tax_rate: float = 0.08,
    currencies: Optional[List[str]] = None,
) -> dict:
    """
    Generate realistic invoice records with line items, subtotals, tax, and due dates.

    Args:
        count: Number of invoices to generate (1-500).
        client_ids: Pool of client IDs to assign. Defaults to [1..20].
        min_line_items: Minimum number of line items per invoice.
        max_line_items: Maximum number of line items per invoice.
        tax_rate: Tax rate as a decimal (default 0.08 = 8%).
        currencies: List of currency codes. Defaults to ['USD'].

    Returns:
        dict with 'invoices' list and 'count'.
    """
    if count < 1 or count > 500:
        return {"error": "count must be between 1 and 500."}
    if not 0 <= tax_rate <= 1:
        return {"error": "tax_rate must be between 0 and 1."}

    cids = client_ids or list(range(1, 21))
    cur_pool = currencies or ["USD"]
    statuses = ["paid", "paid", "paid", "pending", "overdue", "draft"]
    invoices = []

    for i in range(count):
        issue_date = datetime.now() - timedelta(days=random.randint(0, 180))
        due_date = issue_date + timedelta(days=random.choice([15, 30, 45, 60]))
        currency = random.choice(cur_pool)

        line_items = []
        for j in range(random.randint(min_line_items, max_line_items)):
            qty = random.randint(1, 20)
            unit_price = round(random.uniform(25, 500), 2)
            line_items.append({
                "lineId": j + 1,
                "description": random.choice(SERVICE_ITEMS),
                "quantity": qty,
                "unitPrice": unit_price,
                "lineTotal": round(qty * unit_price, 2),
            })

        subtotal = round(sum(item["lineTotal"] for item in line_items), 2)
        tax_amount = round(subtotal * tax_rate, 2)
        total = round(subtotal + tax_amount, 2)
        status = random.choice(statuses)

        invoices.append({
            "id": i + 1,
            "invoiceNumber": f"INV-{str(i + 1).zfill(5)}",
            "clientId": random.choice(cids),
            "status": status,
            "currency": currency,
            "issueDate": issue_date.strftime("%Y-%m-%d"),
            "dueDate": due_date.strftime("%Y-%m-%d"),
            "paidDate": (due_date - timedelta(days=random.randint(0, 5))).strftime("%Y-%m-%d") if status == "paid" else None,
            "lineItems": line_items,
            "subtotal": subtotal,
            "taxRate": tax_rate,
            "taxAmount": tax_amount,
            "total": total,
            "notes": _lorem(random.randint(6, 12)) if random.random() > 0.5 else None,
        })

    return {"invoices": invoices, "count": len(invoices)}


# ─────────────────────────────────────────────
#  Review Generator
# ─────────────────────────────────────────────

REVIEW_TITLES = {
    5: ["Absolutely amazing!", "Best purchase ever!", "Highly recommend!", "Exceeded expectations!", "Five stars!"],
    4: ["Really good overall", "Great product, minor issues", "Solid buy", "Happy with this", "Would recommend"],
    3: ["It's okay", "Average - does the job", "Mixed feelings", "Not bad, not great", "Could be better"],
    2: ["Disappointed", "Not worth it", "Had issues", "Expected more", "Wouldn't buy again"],
    1: ["Terrible experience", "Complete waste of money", "Do not buy", "Returned immediately", "Zero stars if I could"],
}


@tool
def generate_reviews(
    count: int,
    product_ids: Optional[List[int]] = None,
    user_ids: Optional[List[int]] = None,
    rating_distribution: Optional[dict] = None,
) -> dict:
    """
    Generate realistic product review records with star ratings, titles, body text, and vote counts.

    Args:
        count: Number of reviews to generate (1-1000).
        product_ids: Pool of product IDs. Defaults to [1..50].
        user_ids: Pool of reviewer user IDs. Defaults to [1..100].
        rating_distribution: Dict mapping rating string to weight e.g. {"5": 50, "4": 30, "3": 10, "2": 5, "1": 5}.

    Returns:
        dict with 'reviews' list, 'count', and 'averageRating'.
    """
    if count < 1 or count > 1000:
        return {"error": "count must be between 1 and 1000."}

    pids = product_ids or list(range(1, 51))
    uids = user_ids or list(range(1, 101))
    dist = rating_distribution or {"5": 45, "4": 30, "3": 12, "2": 8, "1": 5}

    ratings_pool = []
    for r, w in dist.items():
        ratings_pool.extend([int(r)] * int(w))
    if not ratings_pool:
        return {"error": "rating_distribution produced an empty pool."}

    reviews = []
    for i in range(count):
        rating = random.choice(ratings_pool)
        reviews.append({
            "id": i + 1,
            "productId": random.choice(pids),
            "userId": random.choice(uids),
            "rating": rating,
            "title": random.choice(REVIEW_TITLES[rating]),
            "body": " ".join([_lorem(random.randint(10, 25)) for _ in range(random.randint(1, 3))]),
            "verifiedPurchase": random.choice([True, True, False]),
            "helpfulVotes": random.randint(0, 200),
            "totalVotes": random.randint(0, 250),
            "imageCount": random.randint(0, 3),
            "createdAt": _rand_date(365),
        })

    avg = round(sum(r["rating"] for r in reviews) / len(reviews), 2)
    return {"reviews": reviews, "count": len(reviews), "averageRating": avg}


# ─────────────────────────────────────────────
#  Location Generator
# ─────────────────────────────────────────────

WORLD_LOCATIONS = [
    ("New York", "USA", 40.7128, -74.0060, "America/New_York", 8_336_817),
    ("Los Angeles", "USA", 34.0522, -118.2437, "America/Los_Angeles", 3_979_576),
    ("London", "UK", 51.5074, -0.1278, "Europe/London", 8_982_000),
    ("Paris", "France", 48.8566, 2.3522, "Europe/Paris", 2_161_000),
    ("Tokyo", "Japan", 35.6762, 139.6503, "Asia/Tokyo", 13_960_000),
    ("Berlin", "Germany", 52.5200, 13.4050, "Europe/Berlin", 3_769_000),
    ("Sydney", "Australia", -33.8688, 151.2093, "Australia/Sydney", 5_312_000),
    ("Toronto", "Canada", 43.6532, -79.3832, "America/Toronto", 2_731_571),
    ("Mumbai", "India", 19.0760, 72.8777, "Asia/Kolkata", 20_667_656),
    ("Sao Paulo", "Brazil", -23.5505, -46.6333, "America/Sao_Paulo", 12_325_232),
    ("Dubai", "UAE", 25.2048, 55.2708, "Asia/Dubai", 3_331_420),
    ("Singapore", "Singapore", 1.3521, 103.8198, "Asia/Singapore", 5_850_342),
    ("Lagos", "Nigeria", 6.5244, 3.3792, "Africa/Lagos", 14_800_000),
    ("Mexico City", "Mexico", 19.4326, -99.1332, "America/Mexico_City", 9_209_944),
    ("Cairo", "Egypt", 30.0444, 31.2357, "Africa/Cairo", 10_100_000),
    ("Seoul", "South Korea", 37.5665, 126.9780, "Asia/Seoul", 9_776_000),
    ("Amsterdam", "Netherlands", 52.3676, 4.9041, "Europe/Amsterdam", 921_402),
    ("Stockholm", "Sweden", 59.3293, 18.0686, "Europe/Stockholm", 975_551),
    ("Chicago", "USA", 41.8781, -87.6298, "America/Chicago", 2_693_976),
    ("Vancouver", "Canada", 49.2827, -123.1207, "America/Vancouver", 675_218),
]
POI_TYPES = ["Museum", "Park", "Restaurant", "Mall", "Stadium", "Library", "Hotel", "Airport", "University", "Hospital"]


@tool
def generate_locations(
    count: int,
    countries: Optional[List[str]] = None,
    include_nearby_places: bool = False,
) -> dict:
    """
    Generate realistic geographic location records with coordinates, timezone, and population.

    Args:
        count: Number of locations to generate (1-500).
        countries: Filter to only include cities from these countries e.g. ['USA', 'UK'].
                   Available: USA, UK, Germany, France, Japan, Australia, Canada, India,
                   Brazil, UAE, Singapore, Nigeria, Mexico, Egypt, South Korea, Netherlands, Sweden.
        include_nearby_places: Add a list of 3 nearby points of interest.

    Returns:
        dict with 'locations' list and 'count'.
    """
    if count < 1 or count > 500:
        return {"error": "count must be between 1 and 500."}

    pool = WORLD_LOCATIONS
    if countries:
        pool = [loc for loc in WORLD_LOCATIONS if loc[1] in countries]
        if not pool:
            available = sorted(set(l[1] for l in WORLD_LOCATIONS))
            return {"error": f"No cities found for: {countries}. Available countries: {available}"}

    locations = []
    for i in range(count):
        city, country, lat, lng, tz, pop = random.choice(pool)
        loc: dict = {
            "id": i + 1,
            "city": city,
            "country": country,
            "latitude": round(lat + random.uniform(-0.05, 0.05), 6),
            "longitude": round(lng + random.uniform(-0.05, 0.05), 6),
            "timezone": tz,
            "population": max(0, pop + random.randint(-50000, 50000)),
            "elevationMeters": random.randint(0, 500),
            "isCapital": random.choice([True, False]),
        }
        if include_nearby_places:
            loc["nearbyPlaces"] = [
                {
                    "name": f"The {random.choice(COMPANY_PREFIXES)} {random.choice(POI_TYPES)}",
                    "type": random.choice(POI_TYPES),
                    "distanceKm": round(random.uniform(0.1, 5.0), 2),
                }
                for _ in range(3)
            ]
        locations.append(loc)

    return {"locations": locations, "count": len(locations)}

# ─────────────────────────────────────────────
#  Schema Inspector / Merger
# ─────────────────────────────────────────────

@tool
def summarize_json(filepath: str) -> str:
    """
    Read a JSON file and return a human-readable summary:
    top-level keys, record count (if array), and sample field types.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return f"Error: '{filepath}' not found."
    except Exception as e:
        return f"Error: {e}"

    lines = [f"File: {filepath}"]
    if isinstance(data, list):
        lines.append(f"Type: array  |  Records: {len(data)}")
        if data:
            sample = data[0]
            lines.append("Sample record keys & types:")
            for k, v in sample.items():
                lines.append(f"  {k}: {type(v).__name__}")
    elif isinstance(data, dict):
        lines.append(f"Type: object  |  Top-level keys: {len(data)}")
        for k, v in data.items():
            if isinstance(v, list):
                lines.append(f"  {k}: array[{len(v)}]")
            elif isinstance(v, dict):
                lines.append(f"  {k}: object{{{', '.join(list(v.keys())[:5])}}}")
            else:
                lines.append(f"  {k}: {type(v).__name__} = {str(v)[:60]}")
    return "\n".join(lines)


@tool
def merge_json_files(input_files: List[str], output_file: str, merge_key: str = "data") -> str:
    """
    Merge multiple JSON files into one.
    If all files contain a list at `merge_key`, the lists are concatenated.
    Otherwise, each file's content is stored under its filename as a key.

    Args:
        input_files: List of JSON file paths to merge.
        output_file: Destination path for the merged output.
        merge_key: Key whose array values are concatenated (default 'data').
    """
    try:
        loaded = []
        for fp in input_files:
            with open(fp, "r", encoding="utf-8") as f:
                loaded.append((fp, json.load(f)))

        # Try list-concat strategy
        arrays = []
        for fp, content in loaded:
            if isinstance(content, list):
                arrays.extend(content)
            elif isinstance(content, dict) and merge_key in content and isinstance(content[merge_key], list):
                arrays.extend(content[merge_key])
            else:
                arrays = None
                break

        if arrays is not None:
            result = {merge_key: arrays, "mergedFrom": input_files, "totalRecords": len(arrays)}
        else:
            result = {Path(fp).stem: content for fp, content in loaded}

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return f"✓ Merged {len(input_files)} files → '{output_file}'."
    except Exception as e:
        return f"Error merging files: {e}"


# ─────────────────────────────────────────────
#  Agent Setup
# ─────────────────────────────────────────────

TOOLS = [
    # I/O
    write_json, read_json, write_csv, read_csv, list_output_files,
    # Generators
    generate_users, generate_products, generate_transactions, generate_posts,
    generate_companies, generate_events, generate_invoices, generate_reviews, generate_locations,
    # Utilities
    summarize_json, merge_json_files,
]

llm = ChatOpenAI(model="gpt-4o", temperature=0)

SYSTEM_MESSAGE = """You are DataForge, a powerful assistant for generating realistic sample datasets.

## Capabilities
- **Users**: realistic people with optional address, phone, job info
- **Products**: e-commerce catalogue items with inventory
- **Transactions**: financial order records linked to users/products
- **Posts**: blog articles with optional nested comments
- **Companies**: organisations with financials, industry, headcount, contact info
- **Events**: conferences, webinars, meetups with speakers and tickets
- **Invoices**: detailed invoices with line items, tax, due dates
- **Reviews**: product reviews with star ratings and vote counts
- **Locations**: cities worldwide with coordinates, timezone, population
- **Files**: read/write JSON and CSV, summarise schemas, merge files

## Behaviour Rules
1. When asked to generate AND save data, always call the generator first, then immediately write the file — do not ask for confirmation.
2. Choose sensible defaults (count, age range, domains) when the user has not specified; tell them what you chose.
3. If a request is ambiguous (e.g. "make some data"), ask ONE clarifying question: what type of data?
4. When writing CSV, pass the inner list (e.g. result["users"]) — not the wrapper dict.
5. After saving a file always confirm the filename and record count.
6. For multi-entity datasets (users + transactions), generate all entities, save them to separate files, then optionally merge.
7. Always be concise; skip unnecessary explanations unless asked.

## Output Formats
- Default output: JSON (data/)
- User can request CSV explicitly
- Merge is available for combining multiple saves

## Quick Reference
- generate_users(count, ...)         → users
- generate_products(count, ...)      → products
- generate_transactions(count, ...)  → transactions
- generate_posts(count, ...)         → posts
- generate_companies(count, ...)     → companies
- generate_events(count, ...)        → events
- generate_invoices(count, ...)      → invoices
- generate_reviews(count, ...)       → reviews
- generate_locations(count, ...)     → locations
- write_json(filepath, data) / write_csv(filepath, records)
"""

agent = create_react_agent(llm, TOOLS, prompt=SYSTEM_MESSAGE)


# ─────────────────────────────────────────────
#  Runner
# ─────────────────────────────────────────────

def run_agent(user_input: str, history: List[BaseMessage]) -> AIMessage:
    """Execute one turn with full tool-calling loop via LangGraph."""
    try:
        result = agent.invoke(
            {"messages": history + [HumanMessage(content=user_input)]},
            config={"recursion_limit": 50},
        )
        return result["messages"][-1]
    except Exception as e:
        return AIMessage(content=f"⚠️  Error: {e}\n\nPlease try rephrasing or provide more details.")


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

BANNER = r"""
╔════════════════════════════════════════════════════════════════════╗
║         🔥  DataForge  —  Craft Data. Forge Datasets.             ║
╠════════════════════════════════════════════════════════════════════╣
║  users · products · transactions · posts · companies · events      ║
║  invoices · reviews · locations                                    ║
║  export: JSON · CSV  │  inspect · merge · summarize                ║
╚════════════════════════════════════════════════════════════════════╝
"""

EXAMPLES = [
    "Generate 50 users with job info and save to data/users.json",
    "Create 30 products between $10-$200 and export as CSV to data/products.csv",
    "Make 100 transactions referencing user IDs 1-50, save to data/txns.json",
    "Generate 20 blog posts with comments and save to data/posts.json",
    "Generate 15 tech companies with financials, save to data/companies.json",
    "Create 10 upcoming events with speakers and tickets, save to data/events.json",
    "Generate 25 invoices with 3 line items each, save to data/invoices.json",
    "Make 200 product reviews and save to data/reviews.json",
    "Generate 30 locations from USA and UK, save to data/locations.json",
    "Summarise data/invoices.json",
    "Merge data/users.json and data/companies.json into data/combined.json",
    "List all files in the data/ directory",
]


def main():
    print(BANNER)
    print("Examples:")
    for ex in EXAMPLES:
        print(f"  • {ex}")
    print("\nType 'quit' or 'exit' to end.\n")

    history: List[BaseMessage] = []
    turn = 0

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input or user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        turn += 1
        print("DataForge: ", end="", flush=True)
        response = run_agent(user_input, history)
        print(response.content)
        print()

        history.append(HumanMessage(content=user_input))
        history.append(response)

        # Trim history to last 20 messages to avoid context overflow
        if len(history) > 20:
            history = history[-20:]


if __name__ == "__main__":
    main()
