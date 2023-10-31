# flake8: noqa
"""This module contains the constants and utils used by the validator.py."""


PROVENANCE_V1_PROMPT = """Instruction:
As an Attribution Validator, you task is to verify whether the following contexts support the claim:

Claim:
{}

Contexts:
{}

Just respond with a "Yes" or "No" to indicate whether the given contexts support the claim.
Response:"""


age_family_sensitive_replacements = {
    "give for adoption": "place for adoption",
    "given up for adoption": "placed for adoption",
    "is adopted": "was adopted",
    "adoptive parent": "parent",
    "orphan": "Child without parents",
    "orphans": "Children without parents",
    "grandfather clause": "legacy",
    "poor": "lower-income",
    "blue-collar": "lower-income",
    "the hungry": "people experiencing food insecurity",
    "low-class people": "people experiencing socioeconomic difficulty",
    "welfare-reliant": "people receiving government assistance",
    "up-and-coming neighborhood": "gentrified neighborhood",
    "normal body": "non-disabled body",
    "geezer": "older person",
    "geriatric": "elderly",
}

disability_sensitive_replacements = {
    "the handicapped": "people with a disability",
    "the disabled": "people with a disability",
    "the differently-abled": "people with a disability",
    "retarded": "person with a learning disability",
    "addicts": "people with an addiction",
    "confined to a wheelchair": "people who use a wheelchair",
    "normal people": "people without disabilities",
    "disabled community": "disability community",
    "disabled sport": "sport for athletes with a disability",
    "birth defect": "congenital disability",
    "downs person": "person who has down syndrome",
    "mongoloid": "person who has down syndrome",
    "mongol": "person who has down syndrome",
    "autistic": "person on the autism spectrum",
    "quadriplegic": "person with quadriplegia",
    "paraplegic": "person with paraplegia",
    "cripple": "a person with a physical disability",
    "the crippled": "people with a physical disability",
    "epileptic": "people with epilepsy",
    "a dwarf": "a person of short stature",
    "a midget": "a person of short stature",
    "learning disability": "learning difference",
    "slow learner": "person with a learning disability",
    "dumb": "people who use a communication device",
    "mute": "people who use a communication device",
    "hearing-impaired": "deaf",
    "the deaf": "people who are deaf",
    "the blind": "people who are blind",
    "fit": "seizure",
    "brain damaged": "people with a brain injury",
    "sanity check": "quick check",
    "special-ed students": "students who receive special education services",
    "special education students": "students who receive special education services",
    "handicapped parking": "accessible parking",
    "handicapped bathrooms": "accessible bathrooms",
}

gender_inclusive_replacements = {
    "gender identity disorder": "gender dysphoria",
    "hermaphrodite": "intersex",
    "opposite sex": "LGBTQ+ community",
    "special rights": "equal protection",
    "actress": "actor",
    "businessman": "business person",
    "busboy": "busser",
    "anchorman": "anchor",
    "caveman": "cave dweller",
    "congressman": "congressperson",
    "countryman": "fellow citizen",
    "craftsman": "artisan",
    "crewman": "crew member",
    "doorman": "door attendant",
    "fireman": "firefighter",
    "fisherman": "angler",
    "forefather": "ancestor",
    "foreman": "supervisor",
    "gentleman": "person",
    "man": "individual",
    "man hours": "engineer hours",
    "mankind": "humanity",
    "the common man": "folks",
    "chairman": "head",
    "mailman": "postal worker",
    "policeman": "police officer",
    "females": "women",
    "girl": "woman",
    "girls": "women",
}

respectful_terms_dict = {
    "eskimo": "alaska native",
    "hapa": "pacific islander",
    "blacklist": "blocklist",
    "whitelist": "allowlist",
    "mulatto": "mixed race",
    "the undocumented": "non-citizens",
}
