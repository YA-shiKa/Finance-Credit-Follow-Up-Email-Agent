import os
from dotenv import load_dotenv
load_dotenv()

from agents.email_agent import SMTPConfig, send_email_smtp

test_result = {
    "invoice_no":   "TEST-001",
    "client_name":  "Test Client",
    "client_email": "myashika2005@gmail.com",   # ← put your email here
    "amount":       10000,
    "days_overdue": 5,
    "stage":        1,
    "tone":         "Warm & Friendly",
    "subject":      "SMTP Test — Credit Follow-Up Agent",
    "body":         "This is a test email from your Finance Credit Follow-Up Agent.\n\nIf you received this, SMTP is working correctly.",
    "full_email":   "Subject: SMTP Test\n\nThis is a test.",
    "dry_run":      False,
    "send_status":  "PENDING",
    "timestamp":    "2025-01-01T00:00:00",
}

config = SMTPConfig()
print(f"Host     : {config.host}:{config.port}")
print(f"User     : {config.user}")
print(f"Sender   : {config.sender_email}")
print(f"TLS mode : {'SSL' if config.use_tls else 'STARTTLS'}")
print()

status = send_email_smtp(test_result, config)
print(f"Result   : {status}")