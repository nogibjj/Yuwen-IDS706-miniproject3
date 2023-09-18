## Duke IDS 706 Data Engineering mini project 3  
---

**Summary**

The objective of  Week 3 mini project is to create a script utilizing descriptive statistics. This script has been integrated with the Python CiCd automation template: https://github.com/Candice1121/IDS706-template.

---

**Code Location**

You can find the relevant code in the following files:
- `main.py`
- `test_main.py`

---

These functions enable the creation of a simple dataset and the calculation of its descriptive statistics.

**Modification**

1. Update requirements.txt by adding pyarrow and polars modules
2. Update main.py and test_main.py files for Polars Descriptive Statistics Script


**Description**

1. This repo creates this simple data frame using polars
   ![Local Image](/images/dataframe.png)
2. Three descriptive functions are included to test the statistical features of this dataframe and are tested in test_main.py
   ![Local Image](/images/descriptive.png)
3. make lint, test, format
   make lint : Check poorly structured code and stylistic errors.
   make test : It passed tests on function I defined in main.py
   make format : Apply formatting checks
   To make these files, simply type make in your terminal and it will compile files.
   ![Local Image](/images/make.png)
