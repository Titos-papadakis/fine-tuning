"""
Phrase pools and per-subcategory conversation templates.

Separated from the sampling logic so that adapting this PoC to a new domain is
a matter of rewriting one data file, not untangling generation code.

Template slots:
    {product}        product name (also the ground-truth product_name)
    {order_id}       ORD-XXXXX, the ground-truth order_id
    {amount}         the ground-truth disputed/charged amount
    {plan_amount}    DECOY amount -- an incidental figure the model must NOT extract
    {ticket_ref}     DECOY id (TKT-/INV-) that must NOT be read as an order_id

Every subcategory has 2 phrasings; combined with tier/repeat/churn/noise/entity
variation this yields far more surface diversity than the sample count, so the
model has to learn the mapping rather than memorise a template.
"""
from __future__ import annotations

PRODUCTS = [
    "CloudSync Pro", "DataVault Enterprise", "SwiftShip API", "PayFlow Gateway",
    "InsightBoard Analytics", "NexaCRM", "StreamLine POS", "VaultKey SSO",
    "PixelForge Suite", "AutoPilot Billing", "OrbitDesk Helpdesk", "LedgerBase Accounting",
]

AGENT_NAMES = ["Alex", "Maria", "Jordan", "Priya", "Tom", "Elena", "Sam", "Nikos"]

GREETINGS = [
    "Hello, thanks for reaching out to {product} support, this is {agent}.",
    "Hi, this is {agent} from {product} support, how can I help?",
    "Good morning, {agent} here from the {product} support desk.",
]

OPENERS = {
    "positive": [
        "Hi there! Quick question, hope you're having a good day.",
        "Hello, I just wanted to check something with you, thanks in advance!",
        "Hey, no big deal but I wanted to ask about something.",
    ],
    "neutral": [
        "Hi, I have an issue I'd like some help with.",
        "Hello, I'm reaching out about a problem I ran into.",
        "Hi, could you help me with something on my account?",
    ],
    "negative": [
        "Hi, I'm not happy about something and need this sorted out.",
        "Hello, I've got a problem that really shouldn't have happened.",
        "Hi. I'm quite disappointed with how this has gone.",
    ],
    "frustrated": [
        "This is honestly ridiculous, I need this fixed right now.",
        "I'm extremely frustrated, this has been going on for days and nobody is helping.",
        "Okay I've had enough of this, someone needs to actually fix it.",
    ],
}

CLOSERS = {
    "positive": ["Thanks so much for the quick help!", "Great, appreciate it, have a good one!"],
    "neutral": ["Okay, thanks for looking into it.", "Alright, let me know once it's done."],
    "negative": ["I hope this actually gets resolved this time.", "Please make sure this doesn't happen again."],
    "frustrated": ["This is the last time I chase this up.", "I want this escalated, properly."],
}

# --- Rubric signal surface forms (deliberately scattered and varied) ---------

TIER_MENTIONS = {
    "free": [
        "Customer: Just so you know I'm only on the free plan, if that matters.",
        "Agent: I can see you're currently on our Free tier.",
    ],
    "pro": [
        "Customer: We've been on the Pro plan for about two years now.",
        "Agent: Thanks, I can see your account is on the Pro plan.",
    ],
    "enterprise": [
        "Customer: We're an Enterprise customer, this affects our whole team.",
        "Agent: I see you're on our Enterprise plan, let me prioritise this accordingly.",
        "Customer: Our company signed the Enterprise contract last quarter.",
    ],
}

REPEAT_CONTACT_LINES = [
    "Customer: This is the third time I'm contacting you about this, by the way.",
    "Customer: I already raised this last week and nothing happened.",
    "Customer: I've been back and forth with your team on this for two weeks now.",
]

CHURN_LINES = [
    "Customer: If this isn't fixed today I'm cancelling our subscription.",
    "Customer: Honestly we're already looking at competitors because of this.",
    "Customer: At this point I'm ready to close the account entirely.",
]

DATA_LOSS_LINES = [
    "Customer: And some of our records seem to have disappeared completely.",
    "Customer: I think we've permanently lost the data from last month.",
]

# Neutral filler that adds realistic length without adding signal.
FILLER_TURNS = [
    "Agent: Bear with me one moment while I pull that up.",
    "Customer: Sure, no problem.",
    "Agent: Thanks for waiting.",
    "Customer: Okay, I'm still here.",
    "Agent: Just checking one more thing on my side.",
]

# Decoy turns: contain numbers/ids that must NOT end up in extracted_entities.
DISTRACTOR_TURNS = [
    "Agent: For your reference, your support ticket number is {ticket_ref}.",
    "Customer: I called your line on 555-0143 earlier and got nowhere.",
    "Agent: I'm noting this under internal case {ticket_ref}.",
    "Customer: My account number ends in 4471 if that helps.",
]

RESOLUTION_LINES = {
    "resolved": [
        "Agent: Great, this has now been fully resolved on our end.",
        "Agent: All done, I've completed that for you just now.",
    ],
    "pending": [
        "Agent: I've submitted this and we'll follow up as soon as we hear back.",
        "Agent: It's in progress, you should hear back within a couple of days.",
    ],
    "escalated": [
        "Agent: I'm escalating this to a specialist team who will reach out shortly.",
        "Agent: I've passed this to the team who own this area, they'll take it from here.",
    ],
    "unresolved": [
        "Agent: I don't have a solution yet, but I've logged this for further investigation.",
        "Agent: I'm afraid I can't resolve this from my side right now.",
    ],
}

# --- Per-subcategory issue templates ----------------------------------------
# Each entry: list of (customer_line, agent_probe_line) variants.

ISSUE_TEMPLATES: dict[str, list] = {
    # billing
    "duplicate_charge": [
        ("I was charged twice for my {product} subscription, order {order_id} — two separate ${plan_amount} charges landed on the same card this week, ${amount} in total.",
         "Agent: I can see two identical charges on your account, let me pull up the transaction log."),
        ("There are two identical {product} charges on my card for order {order_id}, back to back — ${plan_amount} each, ${amount} altogether.",
         "Agent: Let me check the transaction history for that order."),
    ],
    "incorrect_invoice": [
        ("My latest {product} invoice has one line item but the number on it is wrong: it shows ${amount} when my plan rate is ${plan_amount}.",
         "Agent: Let me check your billing history and current plan tier."),
        ("The single {product} invoice I just received lists ${amount}. That's not the ${plan_amount} rate we agreed on — the invoice itself has the wrong figure.",
         "Agent: I'll compare that against your contracted rate."),
    ],
    "subscription_renewal": [
        ("My {product} subscription auto-renewed for ${amount} and I wanted to cancel before that happened.",
         "Agent: I understand, let's see what options we have for this renewal."),
        ("You renewed my {product} plan for another year at ${amount} without asking me.",
         "Agent: Let me look at the renewal settings on your account."),
    ],
    "payment_failed": [
        ("My payment of ${amount} for {product} keeps failing even though my card is valid.",
         "Agent: Let me check the payment gateway logs for that transaction."),
        ("{product} won't take my payment — the ${amount} charge has been declined four times now.",
         "Agent: I'll review the decline codes on our side."),
    ],
    "refund_request": [
        ("I'd like a refund of ${amount} for order {order_id}, I was billed incorrectly.",
         "Agent: I can look into that refund request for you right away."),
        ("Please refund the ${amount} from order {order_id}, that charge shouldn't have happened.",
         "Agent: Let me verify the charge and start a refund for you."),
    ],
    # technical
    "login_issue": [
        ("I can't log into {product} anymore, it keeps saying invalid credentials even after a password reset.",
         "Agent: Let's verify your account status and check for any lockouts."),
        ("{product} won't let me in — every login attempt is rejected even with the right password.",
         "Agent: Let me check the authentication logs for your account."),
    ],
    "app_crash": [
        ("{product} keeps crashing every time I try to open the reports section.",
         "Agent: That sounds like a bug, can you tell me your app version and device?"),
        ("The {product} app closes itself the moment I open reporting.",
         "Agent: Let me see if there's a known crash affecting that screen."),
    ],
    "sync_error": [
        ("Data isn't syncing between our devices in {product}, it's been stuck for hours.",
         "Agent: Let me check the sync service status for your account."),
        ("{product} sync has been frozen since yesterday and nothing is updating.",
         "Agent: I'll look at the sync queue on your workspace."),
    ],
    "performance": [
        ("{product} has been extremely slow all week, pages take forever to load.",
         "Agent: I'll check if there's an ongoing performance incident affecting your region."),
        ("Everything in {product} is crawling — reports take minutes to open.",
         "Agent: Let me review the latency metrics for your account."),
    ],
    "integration_bug": [
        ("The {product} integration with our internal system stopped sending webhook events.",
         "Agent: Let's check your webhook configuration and recent delivery logs."),
        ("Our {product} webhooks have been silent for two days and our pipeline is dead.",
         "Agent: I'll pull the delivery logs for your endpoint."),
    ],
    # shipping
    "delayed_delivery": [
        ("My order {order_id} for {product} was supposed to arrive five days ago and it's still not here.",
         "Agent: Let me check the latest tracking update for that order."),
        ("Order {order_id} is way past its delivery date and I've heard nothing.",
         "Agent: I'll pull up the carrier's latest scan for that shipment."),
    ],
    "lost_package": [
        ("Tracking for order {order_id} hasn't updated in over a week, I think my {product} package is lost.",
         "Agent: I'm sorry to hear that, let's open a trace request with the carrier."),
        ("The {product} shipment on order {order_id} has vanished — no tracking movement at all.",
         "Agent: Let me start a lost-parcel investigation."),
    ],
    "wrong_address": [
        ("I think order {order_id} is being shipped to my old address by mistake.",
         "Agent: Let me pull up the shipping details for that order right away."),
        ("Order {order_id} has the wrong delivery address on it, we moved offices last month.",
         "Agent: I'll check whether we can still redirect that shipment."),
    ],
    "damaged_item": [
        ("The {product} unit from order {order_id} arrived with a cracked case.",
         "Agent: I'm sorry about that, could you send a photo of the damage?"),
        ("My {product} hardware from order {order_id} turned up damaged in the box.",
         "Agent: Let me get a damage claim started for you."),
    ],
    "customs_delay": [
        ("Order {order_id} is stuck in customs and nobody has told me why.",
         "Agent: Let me check the customs clearance status for that shipment."),
        ("My {product} shipment, order {order_id}, has been held at customs for eleven days.",
         "Agent: I'll contact our freight partner about the clearance."),
    ],
    # account
    "password_reset": [
        ("I never received the password reset email for my {product} account — the inbox is just empty, no message at all.",
         "Agent: Let me check if the email is being blocked or delayed."),
        ("The {product} reset link never arrives in my inbox, I've requested it six times and nothing shows up.",
         "Agent: I'll check our mail delivery logs for your address."),
    ],
    "account_locked": [
        ("My {product} account shows a 'locked' notice after a few failed login attempts and I need it unlocked.",
         "Agent: I can help unlock that, let me verify your identity first."),
        ("I'm getting an explicit account-locked message on {product} and can't get any work done until it's lifted.",
         "Agent: Let me confirm a couple of details so I can lift the lock."),
    ],
    "email_change": [
        ("I need to change the email address linked to my {product} account.",
         "Agent: Sure, I'll need to verify a few details before updating that."),
        ("Can you move my {product} login to a different email address?",
         "Agent: Happy to, let me verify ownership of the account first."),
    ],
    "data_deletion_request": [
        ("I'd like to request full deletion of my personal data from {product} under GDPR.",
         "Agent: Understood, I'll file that data deletion request for you."),
        ("Please erase all of my personal data held by {product}, as is my right under GDPR.",
         "Agent: I'll raise a formal erasure request with our privacy team."),
    ],
    "profile_update": [
        ("Some of my company details are outdated in my {product} profile.",
         "Agent: No problem, let's go through the fields that need updating."),
        ("Our billing contact in {product} is wrong, it still lists someone who left.",
         "Agent: I can update those profile fields for you now."),
    ],
    # product
    "missing_feature": [
        ("Does {product} support exporting reports to Excel? I've looked everywhere in the settings and can't find that option — does it just not exist?",
         "Agent: Let me check the current feature set and any workarounds."),
        ("Is there a way to bulk export from {product}? I've searched the whole interface and it doesn't seem to be there at all.",
         "Agent: I'll confirm what export options are available on your plan."),
    ],
    "how_to_use": [
        ("I'm not sure how to set up automated workflows in {product}, could you walk me through it?",
         "Agent: Of course, let me explain the workflow builder step by step."),
        ("How do I configure recurring rules in {product}? The docs weren't clear.",
         "Agent: Happy to walk you through the setup."),
    ],
    "compatibility_question": [
        ("Is {product} compatible with our existing single sign-on provider?",
         "Agent: Let me check our supported SSO integrations list."),
        ("Will {product} work alongside the identity provider we already use?",
         "Agent: I'll confirm which providers we currently support."),
    ],
    "product_defect": [
        ("The reporting dashboard in {product} is showing numbers that don't match our raw data.",
         "Agent: That sounds like a calculation bug, let me escalate this to engineering."),
        ("{product} totals are simply wrong — the dashboard disagrees with the export.",
         "Agent: Let me get engineering to look at that discrepancy."),
    ],
    "feature_request": [
        ("I know {product} doesn't have dark mode today, but it would be great if you added it — is that on the roadmap?",
         "Agent: Thanks for the suggestion, I'll log that as a feature request."),
        ("Could you build scheduled reports into {product}? I understand it's not there yet, just suggesting it — it'd save us hours.",
         "Agent: That's a fair ask, let me record it for the product team."),
    ],
    # refund
    "return_request": [
        ("I want to return my {product} purchase from order {order_id}, it doesn't fit our needs. We paid ${amount}.",
         "Agent: No problem, let me start the return process for that order."),
        ("Please arrange a return for order {order_id} — the ${amount} {product} package isn't right for us.",
         "Agent: I'll set up a return label for that order."),
    ],
    "refund_status": [
        ("I was told my ${amount} refund for order {order_id} would take five days but it's been two weeks.",
         "Agent: Let me check the current status of that refund with our payments team."),
        ("Where is the ${amount} refund for order {order_id}? It was promised a fortnight ago.",
         "Agent: I'll chase that refund with our payments team."),
    ],
    "partial_refund": [
        ("I did receive a refund for order {order_id}, but only ${plan_amount} landed when I was owed the full ${amount} — it's short, not missing.",
         "Agent: Let me look into the breakdown of that refund."),
        ("A refund for order {order_id} arrived, but the amount was short — ${plan_amount} came in instead of the full ${amount}.",
         "Agent: I'll reconcile that refund amount for you."),
    ],
    "warranty_claim": [
        ("My {product} unit from order {order_id} stopped working within the warranty period. It cost ${amount}.",
         "Agent: I'm sorry to hear that, let's file a warranty claim for you."),
        ("The ${amount} {product} hardware from order {order_id} failed after four months.",
         "Agent: That's within warranty, let me open a claim."),
    ],
    "cancellation_refund": [
        ("I cancelled my {product} order {order_id} and nothing has come back at all — zero refund of the ${amount}, not even a partial amount.",
         "Agent: Let me confirm the cancellation and refund timeline for you."),
        ("Order {order_id} was cancelled weeks ago and I haven't received a cent of the ${amount} refund yet.",
         "Agent: I'll verify the cancellation went through properly."),
    ],
    # other
    "general_inquiry": [
        ("I'm evaluating {product} for our company, could you tell me more about enterprise pricing?",
         "Agent: Happy to help, let me share our enterprise plan details."),
        ("We're considering {product} — what do larger plans look like?",
         "Agent: I can walk you through our plan tiers."),
    ],
    "feedback": [
        ("I just wanted to share some feedback on the new {product} interface, it's a big improvement.",
         "Agent: Thank you so much, I'll pass this along to the product team."),
        ("Just a note to say the recent {product} update is really well done.",
         "Agent: That's lovely to hear, I'll share it with the team."),
    ],
    "partnership_request": [
        ("Our company is interested in a reseller partnership for {product}.",
         "Agent: Great to hear, let me connect you with our partnerships team."),
        ("We'd like to discuss reselling {product} in our region.",
         "Agent: I'll put you in touch with the partnerships team."),
    ],
    "press_inquiry": [
        ("I'm a journalist writing about {product} and would like a comment from your team.",
         "Agent: Thanks for reaching out, let me forward this to our press contact."),
        ("I'm working on an article covering {product} and need an official statement.",
         "Agent: I'll route this to our communications team."),
    ],
    "spam_report": [
        ("I keep receiving spam emails claiming to be from {product} support, is this legitimate?",
         "Agent: Thanks for flagging that, it does look like a phishing attempt targeting our brand."),
        ("Someone is sending phishing emails pretending to be {product}. Thought you should know.",
         "Agent: Appreciate the report, I'll pass it to our security team."),
    ],
}

# --- Greek surface forms (~15% of the corpus) --------------------------------
# Compact but genuine: enough for the `language` field to be a real prediction
# and to demonstrate multilingual extraction, without duplicating every template.

EL_GREETINGS = [
    "Agent: Καλησπέρα, είμαι ο/η {agent} από την υποστήριξη του {product}.",
    "Agent: Γεια σας, {agent} από την ομάδα υποστήριξης {product}, πώς μπορώ να βοηθήσω;",
]

EL_OPENERS = {
    "positive": ["Customer: Γεια σας! Μια μικρή ερώτηση έχω, ευχαριστώ εκ των προτέρων."],
    "neutral": ["Customer: Καλησπέρα, έχω ένα θέμα και θα ήθελα βοήθεια."],
    "negative": ["Customer: Γεια σας, δεν είμαι καθόλου ευχαριστημένος με αυτό που έγινε."],
    "frustrated": ["Customer: Αυτό είναι απαράδεκτο, το πρόβλημα συνεχίζεται εδώ και μέρες."],
}

EL_CLOSERS = {
    "positive": ["Customer: Σας ευχαριστώ πολύ για την άμεση βοήθεια!"],
    "neutral": ["Customer: Εντάξει, ευχαριστώ που το κοιτάξατε."],
    "negative": ["Customer: Ελπίζω αυτή τη φορά να λυθεί πραγματικά."],
    "frustrated": ["Customer: Θέλω να γίνει κλιμάκωση, τώρα."],
}

# One entry per subcategory, mirroring ISSUE_TEMPLATES' distinctions. These
# used to be one line per *category*, so a Greek transcript carried no signal
# about its subcategory at all -- the label was a coin flip the text could not
# resolve. Measured on a real run: English 125/125 correct, Greek 6/25, which
# was the entire gap between 87% and 100% record exact.
EL_ISSUE_TEMPLATES: dict[str, list] = {
    # billing
    "duplicate_charge": [
        ("Customer: Χρεώθηκα δύο φορές για τη συνδρομή του {product}, παραγγελία {order_id} — δύο ίδιες χρεώσεις των {plan_amount} ευρώ, {amount} ευρώ συνολικά.",
         "Agent: Βλέπω δύο πανομοιότυπες χρεώσεις, ας ανοίξω το ιστορικό συναλλαγών."),
        ("Customer: Στην κάρτα μου εμφανίζονται δύο ίδιες χρεώσεις του {product} για την παραγγελία {order_id}, {plan_amount} ευρώ η καθεμία.",
         "Agent: Θα ελέγξω τις συναλλαγές για αυτή την παραγγελία."),
    ],
    "incorrect_invoice": [
        ("Customer: Το τελευταίο τιμολόγιο του {product} γράφει {amount} ευρώ, ενώ η τιμή του πακέτου μου είναι {plan_amount} ευρώ — το ποσό στο τιμολόγιο είναι λάθος.",
         "Agent: Ας συγκρίνω το τιμολόγιο με το πακέτο σας."),
        ("Customer: Μου ήρθε τιμολόγιο {product} με ποσό {amount} ευρώ, όχι τα {plan_amount} ευρώ που συμφωνήσαμε.",
         "Agent: Θα ελέγξω την συμφωνημένη τιμή σας."),
    ],
    "subscription_renewal": [
        ("Customer: Η συνδρομή μου στο {product} ανανεώθηκε αυτόματα για {amount} ευρώ, ενώ ήθελα να την ακυρώσω πριν την ανανέωση.",
         "Agent: Κατανοώ, ας δούμε τι επιλογές έχουμε για αυτή την ανανέωση."),
        ("Customer: Ανανεώσατε το πακέτο μου στο {product} για άλλον έναν χρόνο με {amount} ευρώ χωρίς να με ρωτήσετε.",
         "Agent: Θα ελέγξω τις ρυθμίσεις ανανέωσης του λογαριασμού σας."),
    ],
    "payment_failed": [
        ("Customer: Η πληρωμή των {amount} ευρώ για το {product} αποτυγχάνει συνέχεια, ενώ η κάρτα μου είναι έγκυρη.",
         "Agent: Θα ελέγξω τα αρχεία της πύλης πληρωμών."),
        ("Customer: Το {product} δεν δέχεται την πληρωμή μου — η χρέωση των {amount} ευρώ απορρίφθηκε τέσσερις φορές.",
         "Agent: Θα δω τους κωδικούς απόρριψης από την πλευρά μας."),
    ],
    "refund_request": [
        ("Customer: Θα ήθελα επιστροφή {amount} ευρώ για την παραγγελία {order_id}, χρεώθηκα λανθασμένα.",
         "Agent: Μπορώ να εξετάσω αμέσως το αίτημα επιστροφής."),
        ("Customer: Παρακαλώ επιστρέψτε μου τα {amount} ευρώ της παραγγελίας {order_id}, δεν έπρεπε να γίνει αυτή η χρέωση.",
         "Agent: Θα επιβεβαιώσω τη χρέωση και θα ξεκινήσω την επιστροφή."),
    ],
    # technical
    "login_issue": [
        ("Customer: Δεν μπορώ να συνδεθώ στο {product}, λέει λάθος στοιχεία ακόμα και μετά από αλλαγή κωδικού.",
         "Agent: Ας ελέγξουμε την κατάσταση του λογαριασμού σας."),
        ("Customer: Το {product} απορρίπτει κάθε προσπάθεια σύνδεσης, παρότι βάζω τον σωστό κωδικό.",
         "Agent: Θα ελέγξω τα αρχεία ταυτοποίησης του λογαριασμού σας."),
    ],
    "app_crash": [
        ("Customer: Το {product} κλείνει απότομα κάθε φορά που ανοίγω την ενότητα αναφορών.",
         "Agent: Ακούγεται σαν σφάλμα, ποια έκδοση της εφαρμογής έχετε;"),
        ("Customer: Η εφαρμογή {product} κρασάρει μόλις πατήσω στις αναφορές.",
         "Agent: Θα δω αν υπάρχει γνωστό crash σε εκείνη την οθόνη."),
    ],
    "sync_error": [
        ("Customer: Τα δεδομένα δεν συγχρονίζονται μεταξύ των συσκευών μας στο {product}, έχει κολλήσει εδώ και ώρες.",
         "Agent: Θα ελέγξω την υπηρεσία συγχρονισμού για τον λογαριασμό σας."),
        ("Customer: Ο συγχρονισμός στο {product} έχει παγώσει από χθες και τίποτα δεν ενημερώνεται.",
         "Agent: Θα δω την ουρά συγχρονισμού του χώρου εργασίας σας."),
    ],
    "performance": [
        ("Customer: Το {product} είναι εξαιρετικά αργό όλη την εβδομάδα, οι σελίδες αργούν πολύ να φορτώσουν.",
         "Agent: Θα ελέγξω αν υπάρχει πρόβλημα απόδοσης στην περιοχή σας."),
        ("Customer: Όλα στο {product} σέρνονται — οι αναφορές θέλουν λεπτά για να ανοίξουν.",
         "Agent: Θα δω τις μετρήσεις καθυστέρησης του λογαριασμού σας."),
    ],
    "integration_bug": [
        ("Customer: Η ενσωμάτωση του {product} με το εσωτερικό μας σύστημα σταμάτησε να στέλνει webhook events.",
         "Agent: Ας ελέγξουμε τη ρύθμιση των webhooks και τα πρόσφατα αρχεία παράδοσης."),
        ("Customer: Τα webhooks του {product} δεν στέλνουν τίποτα εδώ και δύο μέρες και η ροή μας έχει σταματήσει.",
         "Agent: Θα ανοίξω τα αρχεία παράδοσης για το endpoint σας."),
    ],
    # shipping
    "delayed_delivery": [
        ("Customer: Η παραγγελία {order_id} με το {product} έπρεπε να είχε φτάσει πριν από πέντε μέρες και ακόμα τίποτα.",
         "Agent: Θα ελέγξω την τελευταία ενημέρωση παρακολούθησης."),
        ("Customer: Η παραγγελία {order_id} έχει περάσει πολύ την ημερομηνία παράδοσης και δεν έχω κανένα νέο.",
         "Agent: Θα δω το τελευταίο σκανάρισμα της μεταφορικής."),
    ],
    "lost_package": [
        ("Customer: Η παρακολούθηση της παραγγελίας {order_id} δεν έχει αλλάξει εδώ και μια εβδομάδα, νομίζω ότι το δέμα του {product} χάθηκε.",
         "Agent: Λυπάμαι, ας ανοίξουμε αίτημα αναζήτησης με τη μεταφορική."),
        ("Customer: Η αποστολή του {product} με την παραγγελία {order_id} έχει εξαφανιστεί — καμία κίνηση στην παρακολούθηση.",
         "Agent: Θα ξεκινήσω διερεύνηση για χαμένο δέμα."),
    ],
    "wrong_address": [
        ("Customer: Νομίζω ότι η παραγγελία {order_id} στέλνεται κατά λάθος στην παλιά μου διεύθυνση.",
         "Agent: Θα ανοίξω αμέσως τα στοιχεία αποστολής της παραγγελίας."),
        ("Customer: Η παραγγελία {order_id} έχει λάθος διεύθυνση παράδοσης, μετακομίσαμε γραφεία τον περασμένο μήνα.",
         "Agent: Θα δω αν μπορούμε ακόμα να αλλάξουμε τον προορισμό."),
    ],
    "damaged_item": [
        ("Customer: Η συσκευή {product} από την παραγγελία {order_id} ήρθε με σπασμένο περίβλημα.",
         "Agent: Λυπάμαι γι' αυτό, μπορείτε να στείλετε μια φωτογραφία της ζημιάς;"),
        ("Customer: Ο εξοπλισμός {product} της παραγγελίας {order_id} ήρθε χαλασμένος μέσα στο κουτί.",
         "Agent: Θα ξεκινήσω αίτηση αποζημίωσης για τη ζημιά."),
    ],
    "customs_delay": [
        ("Customer: Η παραγγελία {order_id} έχει κολλήσει στο τελωνείο και κανείς δεν μου λέει γιατί.",
         "Agent: Θα ελέγξω την κατάσταση εκτελωνισμού της αποστολής."),
        ("Customer: Η αποστολή του {product}, παραγγελία {order_id}, κρατείται στο τελωνείο εδώ και έντεκα μέρες.",
         "Agent: Θα επικοινωνήσω με τον συνεργάτη μεταφορών για τον εκτελωνισμό."),
    ],
    # account
    "password_reset": [
        ("Customer: Δεν μου ήρθε ποτέ το email επαναφοράς κωδικού για τον λογαριασμό μου στο {product} — το inbox είναι άδειο.",
         "Agent: Θα ελέγξω αν το email μπλοκάρεται ή καθυστερεί."),
        ("Customer: Ο σύνδεσμος επαναφοράς κωδικού του {product} δεν φτάνει ποτέ, τον ζήτησα έξι φορές.",
         "Agent: Θα δω τα αρχεία αποστολής email για τη διεύθυνσή σας."),
    ],
    "account_locked": [
        ("Customer: Ο λογαριασμός μου στο {product} δείχνει μήνυμα 'κλειδωμένος' μετά από μερικές αποτυχημένες συνδέσεις και θέλω να ξεκλειδωθεί.",
         "Agent: Μπορώ να τον ξεκλειδώσω, πρώτα θα επιβεβαιώσω την ταυτότητά σας."),
        ("Customer: Βλέπω ρητό μήνυμα κλειδώματος λογαριασμού στο {product} και δεν μπορώ να δουλέψω μέχρι να αρθεί.",
         "Agent: Θα επιβεβαιώσω κάποια στοιχεία για να άρω το κλείδωμα."),
    ],
    "email_change": [
        ("Customer: Θέλω να αλλάξω τη διεύθυνση email που είναι συνδεδεμένη με τον λογαριασμό μου στο {product}.",
         "Agent: Βεβαίως, θα χρειαστεί να επιβεβαιώσω κάποια στοιχεία πριν την αλλαγή."),
        ("Customer: Μπορείτε να μεταφέρετε τη σύνδεσή μου στο {product} σε άλλο email;",
         "Agent: Ευχαρίστως, θα επιβεβαιώσω πρώτα ότι ο λογαριασμός είναι δικός σας."),
    ],
    "data_deletion_request": [
        ("Customer: Ζητώ την πλήρη διαγραφή των προσωπικών μου δεδομένων από το {product} βάσει GDPR.",
         "Agent: Κατανοητό, θα καταχωρίσω το αίτημα διαγραφής δεδομένων."),
        ("Customer: Παρακαλώ σβήστε όλα τα προσωπικά μου δεδομένα που κρατά το {product}, είναι δικαίωμά μου κατά τον GDPR.",
         "Agent: Θα ανοίξω επίσημο αίτημα διαγραφής στην ομάδα προστασίας δεδομένων."),
    ],
    "profile_update": [
        ("Customer: Κάποια στοιχεία της εταιρείας μας στο προφίλ του {product} είναι παλιά.",
         "Agent: Κανένα πρόβλημα, ας δούμε ποια πεδία χρειάζονται ενημέρωση."),
        ("Customer: Η επαφή χρέωσης στο {product} είναι λάθος, γράφει ακόμα κάποιον που έφυγε.",
         "Agent: Μπορώ να ενημερώσω τώρα αυτά τα πεδία του προφίλ."),
    ],
    # product
    "missing_feature": [
        ("Customer: Υποστηρίζει το {product} εξαγωγή αναφορών σε Excel; Έψαξα παντού στις ρυθμίσεις και δεν βρίσκω τέτοια επιλογή — μήπως απλώς δεν υπάρχει;",
         "Agent: Θα ελέγξω τις τρέχουσες δυνατότητες και τυχόν εναλλακτικές."),
        ("Customer: Υπάρχει τρόπος για μαζική εξαγωγή από το {product}; Έψαξα όλο το περιβάλλον και δεν φαίνεται να υπάρχει καθόλου.",
         "Agent: Θα επιβεβαιώσω ποιες επιλογές εξαγωγής έχει το πακέτο σας."),
    ],
    "how_to_use": [
        ("Customer: Δεν ξέρω πώς να ρυθμίσω αυτοματοποιημένες ροές εργασίας στο {product}, μπορείτε να με καθοδηγήσετε;",
         "Agent: Φυσικά, θα σας εξηγήσω βήμα βήμα τον δημιουργό ροών."),
        ("Customer: Πώς ρυθμίζω επαναλαμβανόμενους κανόνες στο {product}; Η τεκμηρίωση δεν ήταν σαφής.",
         "Agent: Ευχαρίστως να σας καθοδηγήσω στη ρύθμιση."),
    ],
    "compatibility_question": [
        ("Customer: Είναι το {product} συμβατό με τον πάροχο single sign-on που ήδη χρησιμοποιούμε;",
         "Agent: Θα ελέγξω τη λίστα με τις υποστηριζόμενες ενσωματώσεις SSO."),
        ("Customer: Θα λειτουργεί το {product} μαζί με τον πάροχο ταυτότητας που έχουμε ήδη;",
         "Agent: Θα επιβεβαιώσω ποιους παρόχους υποστηρίζουμε σήμερα."),
    ],
    "product_defect": [
        ("Customer: Ο πίνακας αναφορών στο {product} δείχνει νούμερα που δεν ταιριάζουν με τα πραγματικά μας δεδομένα.",
         "Agent: Ακούγεται σαν σφάλμα υπολογισμού, θα το κλιμακώσω στους μηχανικούς."),
        ("Customer: Τα σύνολα στο {product} είναι απλώς λάθος — ο πίνακας διαφωνεί με την εξαγωγή.",
         "Agent: Θα βάλω τους μηχανικούς να δουν αυτή την απόκλιση."),
    ],
    "feature_request": [
        ("Customer: Ξέρω ότι το {product} δεν έχει dark mode σήμερα, αλλά θα ήταν υπέροχο αν το προσθέτατε — είναι στα σχέδιά σας;",
         "Agent: Ευχαριστούμε για την πρόταση, θα την καταγράψω ως αίτημα λειτουργίας."),
        ("Customer: Θα μπορούσατε να προσθέσετε προγραμματισμένες αναφορές στο {product}; Καταλαβαίνω ότι δεν υπάρχουν ακόμα, απλώς το προτείνω.",
         "Agent: Λογικό αίτημα, θα το καταγράψω για την ομάδα προϊόντος."),
    ],
    # refund
    "return_request": [
        ("Customer: Θέλω να επιστρέψω την αγορά {product} της παραγγελίας {order_id}, δεν καλύπτει τις ανάγκες μας. Πληρώσαμε {amount} ευρώ.",
         "Agent: Κανένα πρόβλημα, θα ξεκινήσω τη διαδικασία επιστροφής."),
        ("Customer: Παρακαλώ κανονίστε επιστροφή για την παραγγελία {order_id} — το πακέτο {product} των {amount} ευρώ δεν μας κάνει.",
         "Agent: Θα ετοιμάσω ετικέτα επιστροφής για την παραγγελία."),
    ],
    "refund_status": [
        ("Customer: Μου είπαν ότι η επιστροφή των {amount} ευρώ για την παραγγελία {order_id} θα έπαιρνε πέντε μέρες, αλλά έχουν περάσει δύο εβδομάδες.",
         "Agent: Θα ελέγξω την τρέχουσα κατάσταση της επιστροφής με το λογιστήριο."),
        ("Customer: Πού είναι η επιστροφή των {amount} ευρώ για την παραγγελία {order_id}; Μου την υποσχέθηκαν πριν από δεκαπέντε μέρες.",
         "Agent: Θα κυνηγήσω αυτή την επιστροφή με το λογιστήριο."),
    ],
    "partial_refund": [
        ("Customer: Πήρα επιστροφή για την παραγγελία {order_id}, αλλά ήρθαν μόνο {plan_amount} ευρώ ενώ μου χρωστάτε ολόκληρα τα {amount} ευρώ — λείπει μέρος, δεν λείπει όλη.",
         "Agent: Θα δω την ανάλυση αυτής της επιστροφής."),
        ("Customer: Ήρθε επιστροφή για την παραγγελία {order_id}, αλλά το ποσό ήταν λιγότερο — {plan_amount} ευρώ αντί για ολόκληρα τα {amount} ευρώ.",
         "Agent: Θα συμφωνήσω το ποσό της επιστροφής για εσάς."),
    ],
    "warranty_claim": [
        ("Customer: Η συσκευή {product} της παραγγελίας {order_id} σταμάτησε να λειτουργεί μέσα στην περίοδο εγγύησης. Κόστισε {amount} ευρώ.",
         "Agent: Λυπάμαι, ας καταχωρίσουμε αίτημα εγγύησης."),
        ("Customer: Ο εξοπλισμός {product} των {amount} ευρώ από την παραγγελία {order_id} χάλασε μετά από τέσσερις μήνες.",
         "Agent: Είναι εντός εγγύησης, θα ανοίξω αίτημα."),
    ],
    "cancellation_refund": [
        ("Customer: Ακύρωσα την παραγγελία {order_id} του {product} και δεν μου έχει επιστραφεί τίποτα — μηδέν από τα {amount} ευρώ, ούτε μέρος τους.",
         "Agent: Θα επιβεβαιώσω την ακύρωση και το χρονοδιάγραμμα της επιστροφής."),
        ("Customer: Η παραγγελία {order_id} ακυρώθηκε πριν από εβδομάδες και δεν έχω λάβει ούτε ένα ευρώ από την επιστροφή των {amount} ευρώ.",
         "Agent: Θα ελέγξω ότι η ακύρωση ολοκληρώθηκε σωστά."),
    ],
    # other
    "general_inquiry": [
        ("Customer: Αξιολογούμε το {product} για την εταιρεία μας, μπορείτε να μου πείτε για τις τιμές enterprise;",
         "Agent: Ευχαρίστως, θα σας στείλω τις λεπτομέρειες του πακέτου enterprise."),
        ("Customer: Σκεφτόμαστε το {product} — πώς είναι τα μεγαλύτερα πακέτα;",
         "Agent: Μπορώ να σας εξηγήσω τις βαθμίδες των πακέτων."),
    ],
    "feedback": [
        ("Customer: Ήθελα απλώς να πω τη γνώμη μου για το νέο περιβάλλον του {product}, είναι μεγάλη βελτίωση.",
         "Agent: Σας ευχαριστούμε πολύ, θα το μεταφέρω στην ομάδα προϊόντος."),
        ("Customer: Ένα σύντομο μήνυμα για να πω ότι η πρόσφατη ενημέρωση του {product} είναι πολύ καλή δουλειά.",
         "Agent: Χαίρομαι που το ακούω, θα το πω στην ομάδα."),
    ],
    "partnership_request": [
        ("Customer: Η εταιρεία μας ενδιαφέρεται για συνεργασία μεταπώλησης του {product}.",
         "Agent: Χαίρομαι, θα σας φέρω σε επαφή με την ομάδα συνεργασιών."),
        ("Customer: Θα θέλαμε να συζητήσουμε τη μεταπώληση του {product} στην περιοχή μας.",
         "Agent: Θα σας συνδέσω με την ομάδα συνεργασιών."),
    ],
    "press_inquiry": [
        ("Customer: Είμαι δημοσιογράφος, γράφω για το {product} και θα ήθελα ένα σχόλιο από την ομάδα σας.",
         "Agent: Ευχαριστούμε, θα το προωθήσω στον υπεύθυνο τύπου."),
        ("Customer: Ετοιμάζω άρθρο για το {product} και χρειάζομαι επίσημη δήλωση.",
         "Agent: Θα το στείλω στην ομάδα επικοινωνίας."),
    ],
    "spam_report": [
        ("Customer: Λαμβάνω συνέχεια spam emails που δήθεν έρχονται από την υποστήριξη του {product}, είναι αληθινά;",
         "Agent: Ευχαριστούμε που το αναφέρατε, μοιάζει με απόπειρα phishing στο όνομά μας."),
        ("Customer: Κάποιος στέλνει phishing emails προσποιούμενος ότι είναι το {product}. Είπα να το ξέρετε.",
         "Agent: Ευχαριστούμε για την αναφορά, θα την προωθήσω στην ομάδα ασφαλείας."),
    ],
}

EL_TIER_MENTIONS = {
    "free": ["Customer: Να ξέρετε ότι είμαι στο δωρεάν πακέτο."],
    "pro": ["Customer: Είμαστε στο πακέτο Pro εδώ και δύο χρόνια."],
    "enterprise": ["Customer: Είμαστε Enterprise πελάτης, επηρεάζεται όλη η ομάδα μας."],
}

EL_REPEAT_LINES = ["Customer: Είναι η τρίτη φορά που επικοινωνώ για το ίδιο θέμα."]
EL_CHURN_LINES = ["Customer: Αν δεν λυθεί σήμερα, θα ακυρώσω τη συνδρομή μας."]
EL_DATA_LOSS_LINES = ["Customer: Και κάποια από τα δεδομένα μας φαίνεται να χάθηκαν οριστικά."]

EL_RESOLUTION_LINES = {
    "resolved": ["Agent: Το θέμα επιλύθηκε πλήρως από την πλευρά μας."],
    "pending": ["Agent: Το υπέβαλα και θα επανέλθουμε μόλις έχουμε νεότερα."],
    "escalated": ["Agent: Το κλιμακώνω στην αρμόδια ομάδα, θα επικοινωνήσουν σύντομα."],
    "unresolved": ["Agent: Δεν έχω λύση ακόμα, αλλά το κατέγραψα για περαιτέρω διερεύνηση."],
}

SUBCATEGORY_SUMMARIES = {
    "duplicate_charge": "being billed twice for the same subscription",
    "incorrect_invoice": "an invoice amount not matching the agreed plan price",
    "subscription_renewal": "an unwanted automatic subscription renewal",
    "payment_failed": "repeated payment failures on a valid card",
    "refund_request": "a refund request for an incorrect charge",
    "login_issue": "being unable to log in despite valid credentials",
    "app_crash": "the application crashing in the reporting section",
    "sync_error": "data failing to sync across devices",
    "performance": "persistent slowness across the platform",
    "integration_bug": "a webhook integration failing to deliver events",
    "delayed_delivery": "an order delivery that is significantly delayed",
    "lost_package": "a package with stalled tracking, possibly lost",
    "wrong_address": "an order shipping to an incorrect address",
    "damaged_item": "an item that arrived damaged",
    "customs_delay": "a shipment held up in customs",
    "password_reset": "a password reset email that never arrives",
    "account_locked": "an account locked after failed login attempts",
    "email_change": "a request to change the account email address",
    "data_deletion_request": "a GDPR personal data deletion request",
    "profile_update": "outdated company details on the account profile",
    "missing_feature": "a question about an export feature",
    "how_to_use": "guidance requested on configuring workflows",
    "compatibility_question": "a compatibility question about SSO providers",
    "product_defect": "a dashboard reporting figures that disagree with raw data",
    "feature_request": "a suggestion for a new product feature",
    "return_request": "a request to return a purchase",
    "refund_status": "a refund taking far longer than promised",
    "partial_refund": "a refund that arrived only partially",
    "warranty_claim": "a product failure within the warranty period",
    "cancellation_refund": "a refund still pending after an order cancellation",
    "general_inquiry": "an enterprise pricing inquiry",
    "feedback": "positive feedback about a recent redesign",
    "partnership_request": "a reseller partnership inquiry",
    "press_inquiry": "a press inquiry from a journalist",
    "spam_report": "a report of phishing emails impersonating the brand",
}

CATEGORY_SUBCATS = {
    "billing": ["duplicate_charge", "incorrect_invoice", "subscription_renewal", "payment_failed", "refund_request"],
    "technical": ["login_issue", "app_crash", "sync_error", "performance", "integration_bug"],
    "shipping": ["delayed_delivery", "lost_package", "wrong_address", "damaged_item", "customs_delay"],
    "account": ["password_reset", "account_locked", "email_change", "data_deletion_request", "profile_update"],
    "product": ["missing_feature", "how_to_use", "compatibility_question", "product_defect", "feature_request"],
    "refund": ["return_request", "refund_status", "partial_refund", "warranty_claim", "cancellation_refund"],
    "other": ["general_inquiry", "feedback", "partnership_request", "press_inquiry", "spam_report"],
}

# Which categories reference an order id / a monetary amount at all.
NEEDS_ORDER = {"billing", "shipping", "refund"}
NEEDS_AMOUNT = {"billing", "refund"}

SENTIMENT_WEIGHTS = {
    "billing": {"negative": 0.40, "frustrated": 0.30, "neutral": 0.25, "positive": 0.05},
    "technical": {"negative": 0.35, "frustrated": 0.35, "neutral": 0.25, "positive": 0.05},
    "shipping": {"negative": 0.35, "frustrated": 0.25, "neutral": 0.30, "positive": 0.10},
    "account": {"neutral": 0.45, "negative": 0.25, "frustrated": 0.15, "positive": 0.15},
    "product": {"neutral": 0.50, "positive": 0.25, "negative": 0.20, "frustrated": 0.05},
    "refund": {"negative": 0.35, "frustrated": 0.30, "neutral": 0.30, "positive": 0.05},
    "other": {"positive": 0.40, "neutral": 0.45, "negative": 0.10, "frustrated": 0.05},
}

# --- Actions -----------------------------------------------------------------
# Every action label is paired with the agent utterance that states it, and the
# utterance is rendered into the transcript. A label that is never spoken in the
# conversation would be unpredictable by construction -- pure label noise that
# silently caps the achievable score and makes the benchmark meaningless.
# Format: category -> [(label, agent_utterance), ...]

ACTION_UTTERANCES = {
    "billing": [
        ("Verified customer identity", "Agent: I've verified your identity on the account."),
        ("Reviewed billing history", "Agent: I've gone through your full billing history."),
        ("Issued a refund for the disputed charge", "Agent: I've issued a refund for the disputed charge."),
        ("Applied an account credit", "Agent: I've applied a credit to your account."),
        ("Escalated to the billing team", "Agent: I've escalated this to our billing team."),
        ("Corrected the invoice", "Agent: I've corrected the invoice on our side."),
        ("Updated the payment method on file", "Agent: I've updated the payment method we have on file."),
    ],
    "technical": [
        ("Reproduced the issue internally", "Agent: I've managed to reproduce the issue internally."),
        ("Opened a bug ticket for engineering", "Agent: I've opened a bug ticket for our engineering team."),
        ("Provided a temporary workaround", "Agent: I've sent you a temporary workaround to use meanwhile."),
        ("Cleared cache and reset the session", "Agent: I've cleared the cache and reset your session."),
        ("Escalated to the technical team", "Agent: I've escalated this to our technical team."),
        ("Rolled back a recent configuration change", "Agent: I've rolled back a recent configuration change."),
        ("Advised customer to update to the latest version", "Agent: I'd advise updating to the latest version."),
    ],
    "shipping": [
        ("Opened a trace request with the carrier", "Agent: I've opened a trace request with the carrier."),
        ("Issued a replacement shipment", "Agent: I've issued a replacement shipment for you."),
        ("Updated the shipping address", "Agent: I've updated the shipping address on the order."),
        ("Escalated to the logistics partner", "Agent: I've escalated this to our logistics partner."),
        ("Provided updated tracking information", "Agent: I've sent you updated tracking information."),
        ("Requested photos of the damaged item", "Agent: I've requested photos of the damage from you."),
    ],
    "account": [
        ("Verified customer identity", "Agent: I've verified your identity on the account."),
        ("Reset the account password", "Agent: I've reset the password on your account."),
        ("Unlocked the account", "Agent: I've unlocked your account just now."),
        ("Updated the email on file", "Agent: I've updated the email address we have on file."),
        ("Filed a data deletion request", "Agent: I've filed the data deletion request with our privacy team."),
        ("Updated profile details", "Agent: I've updated those profile details for you."),
    ],
    "product": [
        ("Explained the relevant feature", "Agent: I've explained how that feature works on your plan."),
        ("Shared documentation link", "Agent: I've shared a link to the relevant documentation."),
        ("Logged a feature request", "Agent: I've logged this as a feature request."),
        ("Escalated the defect to engineering", "Agent: I've escalated this defect to engineering."),
        ("Walked customer through the workflow setup", "Agent: I've walked you through the workflow setup."),
    ],
    "refund": [
        ("Started the return process", "Agent: I've started the return process for that order."),
        ("Verified refund eligibility", "Agent: I've verified that this is eligible for a refund."),
        ("Escalated to the payments team", "Agent: I've escalated this to our payments team."),
        ("Issued the outstanding refund balance", "Agent: I've issued the outstanding refund balance."),
        ("Filed a warranty claim", "Agent: I've filed a warranty claim on your behalf."),
        ("Confirmed the cancellation", "Agent: I've confirmed the cancellation went through."),
    ],
    "other": [
        ("Shared enterprise pricing details", "Agent: I've sent over our enterprise pricing details."),
        ("Forwarded feedback to the product team", "Agent: I've forwarded your feedback to the product team."),
        ("Connected customer with the partnerships team", "Agent: I've connected you with our partnerships team."),
        ("Forwarded the inquiry to the press team", "Agent: I've forwarded your inquiry to our press team."),
        ("Flagged the phishing attempt to the security team", "Agent: I've flagged this phishing attempt to our security team."),
    ],
}

# Greek conversations draw from this smaller shared pool so their actions are
# also spoken aloud in-language. Format: (label, en_utterance, el_utterance).
GENERIC_ACTIONS_EL = [
    ("Verified customer identity", "Agent: I've verified your identity on the account.",
     "Agent: Επιβεβαίωσα τα στοιχεία σας στον λογαριασμό."),
    ("Escalated to the technical team", "Agent: I've escalated this to our technical team.",
     "Agent: Το προώθησα στην τεχνική μας ομάδα."),
    ("Shared documentation link", "Agent: I've shared a link to the relevant documentation.",
     "Agent: Σας έστειλα σύνδεσμο με τη σχετική τεκμηρίωση."),
    ("Provided updated tracking information", "Agent: I've sent you updated tracking information.",
     "Agent: Σας έστειλα ενημερωμένες πληροφορίες παρακολούθησης."),
    ("Reviewed billing history", "Agent: I've gone through your full billing history.",
     "Agent: Έλεγξα το ιστορικό χρεώσεών σας."),
    ("Logged a feature request", "Agent: I've logged this as a feature request.",
     "Agent: Κατέγραψα το αίτημα για νέα λειτουργία."),
]
