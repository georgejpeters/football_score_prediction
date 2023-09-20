import json
import time
import ast
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def scrape_fixture_urls(year_passed):
    if year_passed == "2022":
        year = ""
    elif year_passed == "2021":
        year = "-2020-2021"
    link_list = []
    for i in range(1,9):
        if i == 1:
            url = "https://www.oddsportal.com/soccer/england/premier-league"+year+"/results/#/"
        else:
            url = "https://www.oddsportal.com/soccer/england/premier-league"+year+"/results/#/"+"page/"+str(i)+"/"
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        driver = webdriver.Chrome(options=options)
        driver.get(url)
        time.sleep(3)
        page = driver.page_source
        driver.quit()
        soup = BeautifulSoup(page, 'html.parser')
        container = soup.find_all('td', attrs={
            'class': 'name table-participant'})
        for n in container:
            links =n.find_all("a")
            for link in links:
                final_link = "https://www.oddsportal.com"+link["href"]+"#cs;2"
                link_list.append(final_link)
    print(link_list)
    with open("text_files/link_list"+year+".txt", "w") as output:
        output.write(str(link_list))

def scrape_odds_and_results(url):
    #url = "https://www.oddsportal.com/soccer/england/premier-league/manchester-united-brentford-SnxDqmYa/#cs;2"
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    time.sleep(3)
    page = driver.page_source
    driver.quit()
    soup = BeautifulSoup(page, 'html.parser')
    container = soup.find('div', attrs={
        'id': 'odds-data-table'})
    odds_results = container.find_all("a", attrs={"href": ""})
    data = []
    for value in odds_results:
        if value.string != "Compare odds":
            data.append(value.string)
    it = iter(data)
    result_odds_dict = dict(zip(it, it))
    final_ro_dict = {k: v for k, v in result_odds_dict.items() if ":" not in v}
    return final_ro_dict
'''when called crawls oddsportal.com and scrapes odds for correct score predictions'''
def driver_code():
    final_odds_dict = {}
    with open("text_files/link_list.txt", "r") as output:
        l = output.read()
        link_list = ast.literal_eval(l)
    for link in link_list:
        temp_dict = {}
        removed_http = link.rsplit("/", 2)
        teams = removed_http[1].rsplit('-', 1)
        teams_merged = teams[0].replace("-", "")
        odds_dict = scrape_odds_and_results(link)
        temp_dict["team"] = teams_merged
        temp_dict["odds_data"] = odds_dict
        #temp_dict["date"] =
        print(str(teams_merged)+" added")
    print(final_odds_dict)
    with open("json_files/final_odds_dict.json", "w") as fout:
        json.dump(final_odds_dict, fout)
    fout.close()
if __name__ == '__main__':
      scrape_fixture_urls("2021")