import smtplib, time, argparse, logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import mysql.connector
from mysql.connector import Error

# Configuration
SENDER_EMAIL = "hiringdoodle@gmail.com"         
SENDER_APP_PASSWORD_RAW = "kesp mgwu cxlp kicv" 
DB_HOST, DB_PORT, DB_USER, DB_NAME = "localhost", 3306, "root", "hushhush"
DB_PASSWORD = "@darshu123"                     
DB_TABLE = "target_table"                       
SUBJECT = "Coding Assessment Invitation"
SEND_DELAY_SEC = 1.0
ASSESSMENT_LINK = "https://docs.google.com/forms/d/e/1FAIpQLSfOAfeSiECU5_qQ4We6ww5tcyMHrmf_gOXTyHne_6eqv7ejbw/viewform"  # Change this to your real link

# Remove spaces in app password for SMTP
SENDER_APP_PASSWORD = SENDER_APP_PASSWORD_RAW.replace(" ", "").strip()

# mail body
BODY_TEMPLATE = """\
Dear {recipient_name},

You are eligible for our company role. If you are interested, please take the coding assessment using the link below:

{assessment_link}

If you need any assistance, reply to this email.

Thank you,
Hiring Team
"""

def build_message(sender_email, recipient_email, subject, body):
    msg = MIMEMultipart()
    msg["From"], msg["To"], msg["Subject"] = sender_email, recipient_email, subject
    msg.attach(MIMEText(body, "plain"))
    return msg

def send_email_smtp(msg, sender_email, sender_password):
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, [msg["To"]], msg.as_string())
    except smtplib.SMTPAuthenticationError as e:
        raise RuntimeError(
            "Gmail rejected your credentials (SMTP 535). Make sure this is a valid Gmail **App Password**, "
            "2-Step Verification is ON, and the sender email matches the account."
        ) from e

def connect_db():
    return mysql.connector.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD,
        database=DB_NAME, autocommit=True
    )

def fetch_recipients(conn, table):
    q = f"SELECT owner, email FROM {table} WHERE email IS NOT NULL AND email <> ''"
    with conn.cursor(dictionary=True) as cur:
        cur.execute(q)
        return cur.fetchall()

def main():
    parser = argparse.ArgumentParser(description="Send coding assessment invites to all candidates.")
    parser.add_argument("--dry-run", action="store_true", help="Preview recipients without sending emails")
    parser.add_argument("--link", default=ASSESSMENT_LINK, help="Assessment link to include in the email")
    parser.add_argument("--subject", default=SUBJECT, help="Email subject")
    parser.add_argument("--table", default=DB_TABLE, help=f"MySQL table name (default: {DB_TABLE})")
    parser.add_argument("--delay", type=float, default=SEND_DELAY_SEC, help=f"Delay between sends (default: {SEND_DELAY_SEC}s)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        conn = connect_db()
        logging.info("Connected to MySQL.")
    except Error as e:
        raise RuntimeError(f"Failed to connect to MySQL: {e}")

    try:
        rows = fetch_recipients(conn, args.table)
        if not rows:
            logging.warning("No recipients found.")
            return

        logging.info(f"Found {len(rows)} recipients in `{args.table}`.")
        sent = 0

        for r in rows:
            recipient_name = (r.get("owner") or "Candidate").strip()
            recipient_email = (r.get("email") or "").strip()
            if not recipient_email:
                logging.warning(f"Skipping row with empty email (owner={recipient_name})")
                continue

            body = BODY_TEMPLATE.format(
                recipient_name=recipient_name,
                assessment_link=args.link
            )
            msg = build_message(SENDER_EMAIL, recipient_email, args.subject, body)

            if args.dry_run:
                logging.info(f"[DRY RUN] Would send to {recipient_name} <{recipient_email}>")
                continue

            try:
                send_email_smtp(msg, SENDER_EMAIL, SENDER_APP_PASSWORD)
                sent += 1
                logging.info(f"Sent to {recipient_name} <{recipient_email}>")
                time.sleep(args.delay)
            except Exception as e:
                logging.error(f"Failed to send to {recipient_email}: {e}")

        logging.info(f"Done. Emails sent: {sent}/{len(rows)}")
    finally:
        try: conn.close()
        except: pass

if __name__ == "__main__":
    main()
