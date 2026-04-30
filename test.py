# Generate 8-day continuous duty schedule and summarize hours, then export to Excel

from openpyxl import Workbook

names = [
"พลฯชนะโชค สีลานาแก",
"พลฯจักรดุลย์ มีโคดง",
"พลฯฐิติพันธุ์ กองแก้ว",
"พลฯภาคภูมิ พรมมา",
"พลฯบารเมษฐ์ โงะบุตรดา",
"พลฯคงเดช แก้วบับภา",
"พลฯวิษณุพงษ์ ห่มสิงห์"
]

times = [
("06:00-08:00",2),
("08:00-10:00",2),
("10:00-12:00",2),
("12:00-14:00",2),
("14:00-16:00",2),
("16:00-18:00",2),
("18:00-20:00",2),
("20:00-22:00",2),
("22:00-00:00",2),
("00:00-02:00",2),
("02:00-04:00",2),
("04:00-05:30",1.5)
]

days = 8

wb = Workbook()
ws = wb.active
ws.title = "Duty Schedule"

ws.append(["Day","Time","Name","Hours"])

hours_sum = {n:0 for n in names}

shift_counter = 0

for d in range(days):
    for t,h in times:
        name = names[shift_counter % len(names)]
        ws.append([f"Day {d+1}", t, name, h])
        hours_sum[name] += h
        shift_counter += 1

# Summary sheet
ws2 = wb.create_sheet("Summary Hours")
ws2.append(["Name","Total Hours"])

for n,h in hours_sum.items():
    ws2.append([n,h])

path = "duty_schedule_8days.xlsx"
wb.save(path)

hours_sum, path