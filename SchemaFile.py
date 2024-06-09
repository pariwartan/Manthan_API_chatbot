# Define the schema as a dictionary
schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "tax_id": {
      "type": ["string", "null"]
    },
    "company_name": {
      "type": ["string", "null"]
    },
    "acc_total_employees": {
      "type": ["integer", "null"]
    },
    "acc_total_entries": {
      "type": ["integer", "null"]
    },
    "account_checking_total": {
      "type": ["number", "null"]
    },
    "account_number_of_checking": {
      "type": ["integer", "null"]
    },
    "account_number_of_savings": {
      "type": ["integer", "null"]
    },
    "account_savings_total": {
      "type": ["number", "null"]
    },
    "account_total_amount": {
      "type": ["number", "null"]
    },
    "check_date": {
      "type": ["string", "null"],
      "format": "date"
    },
    "company_bank_account": {
      "type": ["string", "null"]
    },
    "company_checkings_total": {
      "type": ["number", "null"]
    },
    "company_number_of_savings": {
      "type": ["integer", "null"]
    },
    "company_saving_total": {
      "type": ["number", "null"]
    },
    "company_total_amount": {
      "type": ["number", "null"]
    },
    "company_total_employees": {
      "type": ["integer", "null"]
    },
    "company_total_entries": {
      "type": ["integer", "null"]
    },
    "end_date": {
      "type": ["string", "null"],
      "format": "date"
    },
    "start_date": {
      "type": ["string", "null"],
      "format": "date"
    },
    "run_date": {
      "type": ["string", "null"],
      "format": "date"
    },
    "company_number_of_checking": {
      "type": ["integer", "null"]
    },
    "direct_deposits": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "checkDate": {
            "type": "string"
          },
          "description": {
            "type": "string"
          },
          "regularHours": {
            "type": "string"
          },
          "overtimeHours": {
            "type": "string"
          },
          "regularAmount": {
            "type": "string"
          },
          "overtimeAmount": {
            "type": "string"
          },
          "totalEarnings": {
            "type": "string"
          },
          "reimburseNOtherPayments": {
            "type": "string"
          },
          "socalSecurityNMedical": {
            "type": "string"
          },
          "federalTax": {
            "type": "string"
          },
          "stateTax": {
            "type": "string"
          },
          "localTax": {
            "type": "string"
          },
          "others": {
            "type": "string"
          },
          "deduction": {
            "type": "string"
          },
          "medical": {
            "type": "string"
          },
          "netpay": {
            "type": "string"
          }
        },
        "required": [
          "checkDate",
          "description",
          "regularHours",
          "overtimeHours",
          "regularAmount",
          "overtimeAmount",
          "totalEarnings",
          "reimburseNOtherPayments",
          "socalSecurityNMedical",
          "federalTax",
          "stateTax",
          "localTax",
          "others",
          "deduction",
          "medical",
          "netpay"
        ]
      }
    }
  },
  "required": [
    "tax_id",
    "company_name",
    "acc_total_employees",
    "acc_total_entries",
    "account_checking_total",
    "account_number_of_checking",
    "account_number_of_savings",
    "account_savings_total",
    "account_total_amount",
    "check_date",
    "company_bank_account",
    "company_checkings_total",
    "company_number_of_savings",
    "company_saving_total",
    "company_total_amount",
    "company_total_employees",
    "company_total_entries",
    "end_date",
    "start_date",
    "run_date",
    "company_number_of_checking",
    "direct_deposits"
  ]
}
