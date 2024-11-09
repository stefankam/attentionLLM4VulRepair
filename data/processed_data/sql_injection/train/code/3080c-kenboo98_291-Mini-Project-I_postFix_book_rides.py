import sqlite3

class Error(Exception):
    pass
class InvalidRNOError(Error):
    pass
class InvalidMemberError(Error):
    pass
class InvalidLocationError(Error):
    pass
class OverbookError(Error):
    pass


class BookRides:
    def __init__(self, cursor):
        self.cursor = cursor
        self.rides = []
        self.rides_dict = dict()
    def find_rides(self, driver):
        <fix/>self.cursor.execute('''</fix>
        SELECT r.rno, r.price, r.rdate, r.seats, r.lugDesc, r.src, r.dst, r.driver, r.cno, r.seats-COUNT(b.bno) 
        FROM rides r, bookings b
        <fix/>WHERE driver = ':driver'</fix>
        AND r.rno = b.bno 
        GROUP BY r.rno, r.price, r.rdate, r.seats, r.lugDesc, r.src, r.dst, r.driver, r.cno
        <fix/>''', {'driver': driver})</fix>
        self.rides = self.cursor.fetchall()

        # create rides dictionary for quick access 
        for ride in self.rides:
            self.rides_dict[ride[0]] = self.rides[1:]
        
    def display_rides(self, page_num):
        page = self.rides[page_num*5: min(page_num*5+5, len(self.rides))]
        for ride in page:
            print(str(ride[0]) + '.', end='')
            print(ride)
        if (page_num*5+5 < len(self.rides)):
            user_input = input("To book a member on a ride, please enter 'b'. To see more rides, please enter 'y'. To exit, press 'e': ")
            if (user_input == 'y'):
                self.display_rides(page_num+1)
        else:
            user_input = input("To book a member on a ride, please enter 'b'. To exit, press 'e': ")
            if (user_input == 'b'):
                self.book_ride()
            else:
                pass
             

    # def find_seats_remaining(self, rno):
    #     query = '''
    #     SELECT r.seats-COUNT(b.bno) FROM rides r, bookings b 
    #     WHERE r.rno = {rno}
    #     AND b.rno = {rno}
    #     '''.format(rno = rno)

    #     self.cursor.execute(query)
    #     rows = self.cursor.fetchone()
    #     return int(rows[0])

    def generate_bno(self):
        query = "SELECT MAX(bno) FROM bookings"
        self.cursor.execute(query)
        max_bno = self.cursor.fetchone()
        return int(max_bno[0])+1

    def verify_email(self, member):
        <fix/>self.cursor.execute("SELECT COUNT(email) FROM members WHERE email = ':email'", {'email':member})</fix>
        result = self.cursor.fetchone()
        if (int(result[0]) > 0):
            return True 
        else:
            return False

    def verify_rno(self, rno):
        <fix/>self.cursor.execute("SELECT COUNT(rno) FROM rides WHERE rno = :rno", {'rno': rno})</fix>
        result = self.cursor.fetchone()
        if (int(result[0]) > 0):
            return True 
        else:
            return False
    
    def verify_location(self, location):
        self.cursor.execute("SELECT COUNT(lccode) FROM locations WHERE lcode = ':lcode'", {'lcode': location})
        return True 
    
    def book_ride(self):

        try:
            rno = input("Please enter a rno: ")
            
            if (not self.verify_rno(rno)):
                raise InvalidRNOError

            member = input("Please enter the email of the member you want to book on the ride: ")

            if (not self.verify_email(member)):
                raise InvalidMemberError

            pickup = input("Please enter pick up location code: ")
            dropoff = input("Please enter pick up location code: ")

            if (not self.verify_location(pickup) or not self.verify_location(dropoff)):
                raise InvalidLocationError

            if (not self.verify_email(member)):
                raise InvalidMemberError

            cost = input("Please enter the cost for ride: ")

            seats = input("Please enter the number of seats for ride: ")

            if (int(seats) > self.rides_dict[rno][-1]):
                overbook = input("Warning: the ride is being over booked, are you sure you want to continue (y/n)")
                if overbook == 'n':
                    raise OverbookError
                else:
                    pass

            #get unique booking number
            bno = self.generate_bno()

            query = '''INSERT INTO bookings VALUES ({bno}, {member}, {rno}, {cost}, {seats}, {pickup}, {dropoff})
                    '''.format(bno = bno, member = member, rno = rno, cost = cost, seats = seats, pickup = pickup, dropoff = dropoff)
            
            print(query)

        except InvalidRNOError:
            print("Please enter a valid rno") 
            self.display_rides(1)
        except InvalidMemberError:
            print("Please enter a valid member email")
            self.display_rides(1)
        except InvalidLocationError:
            print("Please enter a valid pickup and dropoff location code")
            self.display_rides(1)
        except OverbookError:
            print("Please select a fewer number of seats")
            self.display_rides(1)

    
    

