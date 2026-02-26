# VerifyAI — Deploy Guide

## אפשרות 1: Railway (מומלץ — הכי מהיר)

### שלב 1: הכנה
1. היכנס ל-https://railway.app
2. הירשם עם GitHub (חשבון חינם נותן $5 קרדיט)
3. ודא שיש לך חשבון GitHub

### שלב 2: העלאת הקוד ל-GitHub
```bash
cd C:\Users\User\Desktop\verifyai
git init
git add .
git commit -m "VerifyAI v0.1.0 — forensic media authentication"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/verifyai.git
git push -u origin main
```

### שלב 3: Deploy ב-Railway
1. ב-Railway לחץ "New Project" → "Deploy from GitHub Repo"
2. בחר את ה-repo של verifyai
3. Railway יזהה את Procfile באופן אוטומטי
4. לחץ "Deploy" — זה יקח 2-3 דקות

### שלב 4: הגדרת סביבה (אופציונלי)
ב-Railway → Settings → Variables:
- אם רוצה PostgreSQL: הוסף Postgres plugin, ו-Railway יזריק DATABASE_URL
- אם לא — ירוץ על SQLite (מספיק לדמו!)
- OPENAI_API_KEY=sk-xxx (אם יש)

### שלב 5: קבלת URL
Railway ייתן URL כמו: `verifyai-production.up.railway.app`
זה הלינק שתציג למשקיעים!

---

## אפשרות 2: Render (חלופה חינמית)

1. היכנס ל-https://render.com
2. New → Web Service → מ-GitHub
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Free tier — ישן אחרי 15 דקות אבל מתעורר

---

## אפשרות 3: Docker (localhost — כבר עובד)
```bash
docker-compose down -v
docker-compose up --build
# → http://localhost:8000
```

---

## הרצה בלי Docker (לפיתוח)
```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# → http://localhost:8000
# ירוץ על SQLite אוטומטית
```
