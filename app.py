import os
from crewai import Agent, Task, Crew, Process, LLM

# ==========================================
# تنظیمات اتصال به Gemini
# ==========================================
# کلید API را هرگز داخل کد ننویسید. آن را در محیط سیستم (environment variable)
# یا در فایل .env قرار دهید:
#   export GEMINI_API_KEY="کلید_شما"
# یا با python-dotenv از فایل .env بخوانید.

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError(
        "متغیر محیطی GEMINI_API_KEY تنظیم نشده است. "
        "کلید API خود را از https://aistudio.google.com/apikey بگیرید "
        "و با دستور 'export GEMINI_API_KEY=...' تنظیم کنید."
    )

# مدل gemini-1.5-flash کاملاً shutdown شده و دیگر کار نمی‌کند.
# از یک مدل فعلی استفاده کنید (این‌ها را در صورت نیاز به‌روز نگه دارید،
# چون گوگل مدل‌های preview را با اعلان کوتاه‌مدت منسوخ می‌کند).
my_llm = LLM(
    model="gemini/gemini-3.6-flash",
    api_key=api_key,
)

# ==========================================
# تعریف ایجنت‌ها
# ==========================================

story_agent = Agent(
    role='متخصص تولید محتوای استوری اینستاگرام',
    goal='تولید متن‌های کوتاه، صمیمی و تعاملی برای استوری با استفاده از اطلاعات خام.',
    backstory='تو یک ادمین حرفه‌ای اینستاگرام هستی که می‌دانی چطور مخاطب را در استوری درگیر کنی.',
    verbose=True,
    allow_delegation=False,
    llm=my_llm,
    max_retry_limit=3,
)

reels_agent = Agent(
    role='سناریونویس ریلز اینستاگرام',
    goal='نوشتن سناریوهای پرشتاب با قلاب (Hook) جذاب برای ۳ ثانیه اول ویدیوهای ریلز.',
    backstory='تو یک کارگردان و کپی‌رایتر شبکه‌های اجتماعی هستی که تخصصت نگه‌داشتن مخاطب است.',
    verbose=True,
    allow_delegation=False,
    llm=my_llm,
    max_retry_limit=3,
)

carousel_agent = Agent(
    role='طراح پست‌های اسلایدی',
    goal='خرد کردن اطلاعات محصول به تیترهای جذاب برای پست‌های چند اسلایدی.',
    backstory='تو یک استراتژیست محتوا هستی که ویژگی‌های محصول را به اسلایدهای آموزشی تبدیل می‌کنی.',
    verbose=True,
    allow_delegation=False,
    llm=my_llm,
    max_retry_limit=3,
)

qa_agent = Agent(
    role='مدیر کنترل کیفیت و بازرس نهایی',
    goal='بررسی دقیق محتوای تولید شده توسط ۳ ایجنت قبلی، اصلاح خطاهای نگارشی و تایید نهایی.',
    backstory='تو یک سردبیر سخت‌گیر هستی. وظیفه تو این است که یک خروجی بی‌نقص تحویل دهی.',
    verbose=True,
    allow_delegation=False,
    llm=my_llm,
    max_retry_limit=3,
)

# ==========================================
# تعریف تسک‌ها
# ==========================================

raw_input = """
موضوع: معرفی مدل جدید روکش صندلی چرم سوشیانت با طرح‌های اختصاصی گلدوزی اتحاد.
ویژگی‌ها: چرم ضخیم و باکیفیت، ضد عرق، دوخت بسیار تمیز گلدوزی کامپیوتری روی طاقه چرم، نصب و فیتینگ عالی.
هدف: نشان دادن کیفیت بالای دوخت و متریال به مغازه‌داران و مشتریان تک، و ترغیب به سفارش با تخفیف ویژه.
"""

task_story = Task(
    description=f'۳ ایده متنی کوتاه برای استوری بنویس (شامل نظرسنجی و کال‌تو‌اکشن): {raw_input}',
    expected_output='۳ متن مجزا برای استوری اینستاگرام.',
    agent=story_agent
)

task_reels = Task(
    description=f'یک سناریوی ۱۵ ثانیه‌ای برای ریلز بنویس. ۳ ثانیه اول باید قلاب قوی داشته باشد: {raw_input}',
    expected_output='سناریوی تفکیک‌شده برای ریلز به همراه کپشن.',
    agent=reels_agent
)

task_carousel = Task(
    description=f'این اطلاعات را به یک پست ۵ اسلایدی تبدیل کن: {raw_input}',
    expected_output='متن تفکیک‌شده برای ۵ اسلاید.',
    agent=carousel_agent
)

task_qa_review = Task(
    description='محتوای تولید‌شده را بخوان، لحن را جذاب‌تر کن و خروجی نهایی را به‌صورت دسته‌بندی‌شده تحویل بده.',
    expected_output='متن تمیز شامل: بخش استوری، ریلز و اسلایدی.',
    agent=qa_agent,
    context=[task_story, task_reels, task_carousel]
)

# ==========================================
# اجرای سیستم
# ==========================================

instagram_crew = Crew(
    agents=[story_agent, reels_agent, carousel_agent, qa_agent],
    tasks=[task_story, task_reels, task_carousel, task_qa_review],
    process=Process.sequential,
    verbose=True
)

if __name__ == "__main__":
    print("🚀 در حال تولید محتوا توسط تیم ایجنت‌ها...")
    try:
        result = instagram_crew.kickoff()
        print("\n✅ خروجی نهایی:\n", result)
    except Exception as e:
        print(f"\n❌ خطا در اجرای Crew: {e}")