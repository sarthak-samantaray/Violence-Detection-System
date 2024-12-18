import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

class EmailSender:
    def __init__(self, email_address, email_password):
        """
        Initialize the EmailSender with credentials.
        :param email_address: Sender's email address.
        :param email_password: Sender's email password or app-specific password.
        """
        self.email_address = email_address
        self.email_password = email_password

    def send_email(self, recipient_email, subject, body, attachment_path=None):
        """
        Send an email with an optional attachment.
        :param recipient_email: Email address of the recipient.
        :param subject: Subject of the email.
        :param body: Body content of the email.
        :param attachment_path: Path to the attachment (optional).
        :return: None
        """
        # Compose email
        msg = MIMEMultipart()
        msg['From'] = self.email_address
        msg['To'] = recipient_email
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        # Attach a file if provided
        if attachment_path:
            if not os.path.exists(attachment_path):
                raise FileNotFoundError(f"Attachment not found: {attachment_path}")
            with open(attachment_path, 'rb') as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename={os.path.basename(attachment_path)}',
                )
                msg.attach(part)

        # Send the email
        try:
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(self.email_address, self.email_password)
                server.send_message(msg)
        except Exception as e:
            raise RuntimeError(f"Failed to send email: {str(e)}")
