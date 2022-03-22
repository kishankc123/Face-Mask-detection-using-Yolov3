import smtplib, ssl

port = 465  # For SSL
smtp_server = "smtp.gmail.com"
sender_email = "facemaskdetectionandalert@gmail.com "
password = "qahylckwpfzgipnv"


def sendEmail(msg, receiver_email="kishankc017@sxc.edu.np"):
    message = 'Subject: {}\n\n{}'.format("Facemask Notifier", msg)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)

    print("mail sent to", receiver_email)
