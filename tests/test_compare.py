from app.schemas import CnicFields
from app.services.compare import compare_records, parse_date
from app.services.db_client import _normalise
from app.services.ocr import format_cnic, parse_json_fields, regex_fallback

DB = CnicFields(
    name="M Shayan Jameel Mirza",
    father_husband_name="Jameel Khali",
    cnic_number="42101-1389680-1",
    #address="House 12, Street 5, Gulberg III, Lahore",
    date_of_birth="08.07.2000",
    date_of_issue="24.02.2021",
    date_of_expiry="24.02.2031",
)


def test_exact_match():
    assert compare_records(DB.model_copy(), DB).verdict == "match"


def test_ocr_noise_in_name_is_tolerated():
    ext = DB.model_copy(update={"name": "AHMED  RAZA KHAN", "father_husband_name": "Muhammad Aslam Khan."})
    assert compare_records(ext, DB).verdict == "match"


def test_cnic_formatting_ignored_but_digit_mismatch_rejects():
    ok = DB.model_copy(update={"cnic_number": "3520112345671"})
    assert compare_records(ok, DB).verdict == "match"
    bad = DB.model_copy(update={"cnic_number": "35201-1234567-2"})
    assert compare_records(bad, DB).verdict == "reject"


def test_expiry_mismatch_needs_review():
    ext = DB.model_copy(update={"date_of_expiry": "02.06.2032"})
    res = compare_records(ext, DB)
    assert res.verdict == "review" and "date_of_expiry" in res.reason


def test_missing_extracted_field_needs_review():
    ext = DB.model_copy(update={"address": None})
    assert compare_records(ext, DB).verdict == "review"


def test_person_not_found_rejects():
    assert compare_records(DB, None).verdict == "reject"


def test_date_formats():
    assert parse_date("02/06/2021") == parse_date("2021-06-02") == parse_date("02.06.2021")
    assert parse_date("garbage") is None


def test_parse_json_with_fences_and_nulls():
    text = '```json\n{"name": "A B", "cnic_number": "35201 1234567 1", "address": null}\n```'
    f = parse_json_fields(text)
    assert f.name == "A B" and f.address is None
    assert format_cnic(f.cnic_number) == "35201-1234567-1"


def test_regex_fallback_assigns_dates_in_order():
    f = regex_fallback("ID 35201-1234567-1  02.06.2031  14.03.1990  02.06.2021")
    assert f.cnic_number == "35201-1234567-1"
    assert (f.date_of_birth, f.date_of_issue, f.date_of_expiry) == ("14.03.1990", "02.06.2021", "02.06.2031")


def test_db_alias_mapping_includes_allocation_date():
    rec = _normalise({"full_name": "X", "father_name": "Y", "cnic": "1", "allocation_date": "01.01.2020"})
    assert rec.name == "X" and rec.father_husband_name == "Y" and rec.date_of_issue == "01.01.2020"
