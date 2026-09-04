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
        ("I was charged twice for my {product} subscription, order {order_id}. My plan is only ${plan_amount} a month but ${amount} came out in total.",
         "Agent: I can see two identical charges on your account, let me pull up the transaction log."),
        ("There are two {product} charges on my card for order {order_id} — I should have paid ${plan_amount}, not ${amount}.",
         "Agent: Let me check the transaction history for that order."),
    ],
    "incorrect_invoice": [
        ("My latest invoice for {product} shows ${amount} but my plan is supposed to be ${plan_amount}.",
         "Agent: Let me check your billing history and current plan tier."),
        ("The {product} invoice I just received says ${amount}. That's not the ${plan_amount} we agreed on.",
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
        ("I never received the password reset email for my {product} account.",
         "Agent: Let me check if the email is being blocked or delayed."),
        ("The {product} reset link never arrives, I've requested it six times.",
         "Agent: I'll check our mail delivery logs for your address."),
    ],
    "account_locked": [
        ("My {product} account got locked after a few failed login attempts and I need it back.",
         "Agent: I can help unlock that, let me verify your identity first."),
        ("I'm locked out of {product} entirely and can't get any work done.",
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
        ("Does {product} support exporting reports to Excel? I can't find that option.",
         "Agent: Let me check the current feature set and any workarounds."),
        ("Is there a way to bulk export from {product}? I've looked everywhere.",
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
        ("It would be great if {product} supported dark mode, is that on the roadmap?",
         "Agent: Thanks for the suggestion, I'll log that as a feature request."),
        ("Any chance {product} could add scheduled reports? It'd save us hours.",
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
        ("I only got ${plan_amount} back but I was owed ${amount} for order {order_id}.",
         "Agent: Let me look into the breakdown of that refund."),
        ("The refund on order {order_id} was short — ${plan_amount} arrived instead of ${amount}.",
         "Agent: I'll reconcile that refund amount for you."),
    ],
    "warranty_claim": [
        ("My {product} unit from order {order_id} stopped working within the warranty period. It cost ${amount}.",
         "Agent: I'm sorry to hear that, let's file a warranty claim for you."),
        ("The ${amount} {product} hardware from order {order_id} failed after four months.",
         "Agent: That's within warranty, let me open a claim."),
    ],
    "cancellation_refund": [
        ("I cancelled my {product} order {order_id} but haven't seen the ${amount} refund yet.",
         "Agent: Let me confirm the cancellation and refund timeline for you."),
        ("Order {order_id} was cancelled weeks ago and the ${amount} still hasn't come back.",
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

EL_ISSUE_BY_CATEGORY = {
    "billing": ("Customer: Χρεώθηκα {amount} ευρώ για το {product} ενώ το πακέτο μου είναι {plan_amount} ευρώ.",
                "Agent: Ας δούμε το ιστορικό χρεώσεων του λογαριασμού σας."),
    "technical": ("Customer: Το {product} δεν λειτουργεί σωστά εδώ και μέρες, κολλάει συνέχεια.",
                  "Agent: Θα ελέγξω τα τεχνικά αρχεία καταγραφής για τον λογαριασμό σας."),
    "shipping": ("Customer: Η παραγγελία {order_id} με το {product} δεν έχει φτάσει ακόμα.",
                 "Agent: Θα ελέγξω την πορεία της αποστολής σας."),
    "account": ("Customer: Δεν μπορώ να συνδεθώ στον λογαριασμό μου στο {product}.",
                "Agent: Θα ελέγξω την κατάσταση του λογαριασμού σας."),
    "product": ("Customer: Υποστηρίζει το {product} εξαγωγή αναφορών σε Excel;",
                "Agent: Θα ελέγξω τις διαθέσιμες επιλογές εξαγωγής."),
    "refund": ("Customer: Περιμένω επιστροφή {amount} ευρώ για την παραγγελία {order_id} και δεν έχει έρθει.",
               "Agent: Θα ελέγξω την πορεία της επιστροφής χρημάτων."),
    "other": ("Customer: Ενδιαφερόμαστε για το {product} για την εταιρεία μας, θέλουμε πληροφορίες τιμολόγησης.",
              "Agent: Ευχαρίστως, θα σας στείλω τις λεπτομέρειες των πακέτων."),
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
