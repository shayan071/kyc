"""Development-only stand-in for the real database API.

Replace by pointing KYC_DB_API_URL at the real service. Contract used by the app:
    GET {KYC_DB_API_URL}/persons/{cnic_digits}  ->  200 JSON record | 404
"""

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/mock-db", tags=["mock-db"])

# Fictional sample data.
PEOPLE = {
    "3520112345671": {
        "name": "M Shayan Jameel Mirza",
        "father_name": "Jameel Khaliq Mirza",
        "cnic": "42101-1389680-1",
        "dob": "08.07.2001",
        "issue_date": "20.02.2020",
        "expiry_date": "20.02.2030",
    }
}


@router.get("/persons/{cnic}")
def get_person(cnic: str):
    record = PEOPLE.get("".join(c for c in cnic if c.isdigit()))
    if not record:
        raise HTTPException(status_code=404, detail="Person not found")
    return record
