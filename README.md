# 🔥 DataForge

DataForge is an AI-powered CLI tool that generates realistic sample datasets for developers, testers, and data engineers. Just describe what you need in plain English — DataForge handles the rest.

---

## ✨ Features

| Data Type | Fields Generated |
|---|---|
| 👤 **Users** | name, email, username, age, address, phone, job title |
| 📦 **Products** | SKU, name, category, price, stock, warehouse, rating |
| 💳 **Transactions** | order ID, user, product, amount, tax, payment method |
| 📝 **Posts** | title, body, tags, views, likes, nested comments |
| 🏢 **Companies** | name, industry, employees, revenue, funding stage |
| 📅 **Events** | type, speakers, venue, tickets, attendees |
| 🧾 **Invoices** | line items, subtotal, tax, due dates, status |
| ⭐ **Reviews** | star rating, title, body, verified purchase, votes |
| 🌍 **Locations** | city, country, lat/lng, timezone, population |

**Export formats:** JSON · CSV  
**Utilities:** summarize schema · merge files · list outputs

---

## 🚀 Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/dataforge.git
cd dataforge
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install langchain langchain-openai langgraph python-dotenv
```

### 4. Add your OpenAI API key
Create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-your-key-here
```

### 5. Run DataForge
```bash
python3 main.py
```

---

## 💬 Example Commands

```
Generate 50 users with job info and save to data/users.json
Create 30 products between $10-$200 and export as CSV to data/products.csv
Make 100 transactions referencing user IDs 1-50, save to data/txns.json
Generate 20 blog posts with comments and save to data/posts.json
Generate 15 tech companies with financials, save to data/companies.json
Create 10 upcoming events with speakers and tickets, save to data/events.json
Generate 25 invoices with 3 line items each, save to data/invoices.json
Make 200 product reviews and save to data/reviews.json
Generate 30 locations from USA and UK, save to data/locations.json
Summarise data/users.json
Merge data/users.json and data/companies.json into data/combined.json
List all files in the data/ directory
```

---

## 🛠️ How It Works

DataForge uses a **LangGraph ReAct agent** powered by GPT-4o. The agent:

1. Interprets your natural language request
2. Calls the appropriate generator tool(s)
3. Automatically saves the output to a file
4. Confirms what was created

```
You ──▶ Natural language input
         │
         ▼
    GPT-4o (ReAct Agent)
         │
         ▼
    Generator Tools ──▶ JSON / CSV output
```

---

## 📁 Project Structure

```
dataforge/
├── main.py          # Main agent script
├── .env             # Your API key (never commit this!)
├── .gitignore       # Ignores .env and data/ folder
├── README.md        # This file
└── data/            # Generated output files (auto-created)
```

---

## ⚙️ Requirements

- Python 3.10+
- OpenAI API key with billing enabled
- Packages: `langchain`, `langchain-openai`, `langgraph`, `python-dotenv`

---

## 🔒 Security

Never commit your `.env` file. The `.gitignore` in this repo already excludes it. Double-check before pushing:
```bash
git status   # .env should NOT appear here
```

---

## 🗺️ Roadmap

- [ ] Level 2 — Relational data with real foreign keys
- [ ] Level 3 — SQL INSERT export + Excel (.xlsx)
- [ ] Level 4 — Web UI (FastAPI + React)
- [ ] Level 5 — Custom schema input ("describe your database")
- [ ] Level 6 — Direct database push (PostgreSQL, MongoDB)

---
