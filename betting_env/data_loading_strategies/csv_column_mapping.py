class CsvColumnsMappings:
    def __init__(
        self,
        event_id: str,
        event_date: str,
        away_team: str,
        home_team: str,
        league: str,
        away_score: str,
        home_score: str,
        away_odds: str,
        home_odds: str,
        draw_odds: str,
        date_format: str = "%d/%m/%Y",
        column_separator: str = "\t",
    ):
        self.column_separator = column_separator
        self.date_format = date_format
        self.draw_odds = draw_odds
        self.event_id = event_id
        self.event_date = event_date
        self.home_odds = home_odds
        self.away_odds = away_odds
        self.home_score = home_score
        self.away_score = away_score
        self.league = league
        self.home_team = home_team
        self.away_team = away_team
        # datetime.strptime('01/01/2005','%d/%m/%Y')
