culture_events_shown_on_home_page = 10
# If auto_cross_reference = True program will scan description field for url slugs whenever that
# CE is saved. If it finds a matching slug in the description it will add a hyperlink
# If False hyperlinks will only be added to valid slugs within curly brackets {}
auto_cross_reference = True
# A list of allowed HTML tags the user can enter to HTML escaped fields.
bleach_allowed = ['strong', 'p']