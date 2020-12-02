def blackjack_hand_greater_than(hand_1, hand_2):
    """
    Return True if hand_1 beats hand_2, and False otherwise.
    
    In order for hand_1 to beat hand_2 the following must be true:
    - The total of hand_1 must not exceed 21
    - The total of hand_1 must exceed the total of hand_2 OR hand_2's total must exceed 21
    
    Hands are represented as a list of cards. Each card is represented by a string.
    
    When adding up a hand's total, cards with numbers count for that many points. Face
    cards ('J', 'Q', and 'K') are worth 10 points. 'A' can count for 1 or 11.
    
    When determining a hand's total, you should try to count aces in the way that 
    maximizes the hand's total without going over 21. e.g. the total of ['A', 'A', '9'] is 21,
    the total of ['A', 'A', '9', '3'] is 14.
    
    Examples:
    >>> blackjack_hand_greater_than(['K'], ['3', '4'])
    True
    >>> blackjack_hand_greater_than(['K'], ['10'])
    False
    >>> blackjack_hand_greater_than(['K', 'K', '2'], ['3'])
    False
    """
    total1, total2 = 0
    A1_counts = 0
    A2_counts = 0
    for card in hand_1:
        if card == 'A':
            A1_counts += 1
        else:
            if card == 'J' or card == 'Q' or card == 'K':
                total1 += 10
            else:
                total1 += int(card)
    for i in range(A1_counts):
        if total1 <= 10:
            total1 += 11
        else:
            total1 += 1

    for card in hand_2:
        if card == 'A':
            A2_counts += 1
        else:
            if card == 'J' or card == 'Q' or card == 'K':
                total2 += 10
            else:
                total2 += int(card)
    for i in range(A2_counts):
        if total2 <= 10:
            total2 += 11
        else:
            total2 += 1

    if total1 > 21:
        return False
    elif total2 > 21:
        return True
    elif total1 > total2:
        return True
    else:
        return False


blackjack_hand_greater_than(['J', 'A'], ['6'])
