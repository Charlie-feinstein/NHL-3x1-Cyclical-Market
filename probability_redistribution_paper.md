# The Hidden Cycles in Sports Betting Markets
## How Probability Redistribution Creates Exploitable Patterns in 3-Way Hockey Markets

**Whizard Analytics**

*The Edge You Were Not Supposed to Find.*

March 2026

---

## The Pattern That Changed Everything

Picture this: You build a betting model. You backtest it. You see the results plotted on a chart. And there, staring back at you, are these beautiful wave patterns. Your profit line goes up for a couple weeks, then down for a couple weeks, then back up again. It looks almost choreographed, like the market is dancing to some invisible rhythm.

The first time I saw this, I thought it was just noise. Random variance. The natural ups and downs of sports betting. So I built another model. Different sport, different approach, different everything. Same wave patterns.

I built ten more models over the next six years. MLB, NFL, NHL, college football. Expected goals models, machine learning models, traditional statistics models. Every single one showed the same thing: wavy profit lines that oscillated in semi-predictable patterns. But here's the crazy part: I kept ignoring it. I was so focused on building better prediction models that I never stopped to ask the obvious question: What if the waves ARE the opportunity?

Three months ago, I finally asked that question. And what I discovered completely changed how I think about sports betting. But the answer wasn't what I expected. The waves aren't some mysterious force. They're not market manipulation or bookmaker conspiracy. They're the natural, inevitable consequence of a simple mathematical fact that hides in plain sight:

**In a 3-way betting market, probabilities must sum to 100%. Every percentage point the market gives to one outcome, it must take from another.**

That constraint, probability mass conservation, creates cycles. Not because anyone is orchestrating them, but because the market can't help it. When reality temporarily shifts probability toward one outcome, it has to come from somewhere. And when the market is slow to adjust, the imbalance creates exploitable patterns that oscillate over time.

This paper tells that story. I'm going to show you exactly how probability flows between outcomes in the NHL 3-way market, why the market consistently misprices one specific outcome, and how that mispricing creates the wave patterns I'd been staring at for six years without understanding.

No complex math required. Just logic, observation, and 1,593 real games worth of evidence.

---

## Part 1: The Game Nobody Understands

### The Fatal Assumption

Most "analytical bettors" approach sports betting like it's the stock market. They think: if I can predict outcomes better than everyone else, I'll make money. And technically, they're right. But they're missing something huge.

In the stock market, multiple people can win at the same time. If Apple releases an incredible new product and the stock goes up, every shareholder profits together. The market can create wealth. It's expansive. There's theoretically no limit to how much money can be made.

Sports betting is the exact opposite. Every single day, there's a fixed amount of money available to win. Think about it: if there are 10 NHL games tonight, the outcomes of those games are predetermined (we just don't know what they are yet). One side of each bet wins, the other side loses. My profit is your loss. Your profit is someone else's loss. It's a closed system.

But wait, it gets worse. The bookmaker takes a cut of every bet, usually around 4-5%. So it's not even a zero-sum game. It's a negative-sum game. The total amount of money flowing out to winners is LESS than the total amount flowing in from losers. The difference? That's the bookmaker's profit. They're taking their piece off the top, every single day.

This is the single most important thing to understand about sports betting: it's a fixed-profit system with a negative drift. The pie doesn't grow. And after the house takes their cut, what's left for bettors is actually shrinking.

### What This Means for Patterns

Here's where it gets interesting. Because the profit is fixed and bettors cluster around certain types of bets, the market has to redistribute that fixed pot in predictable ways.

But I used to think about this redistribution at the wrong level. I was looking at categories of bets (small edge, medium edge, large edge) and tracking which categories ran hot or cold. That was descriptive, but it wasn't explanatory. I could see the waves, but I couldn't explain WHY they existed at a fundamental level.

The breakthrough came when I started looking at it through the lens of the 3-way market. In NHL betting, there are three possible regulation-time outcomes:

- **Home team wins in regulation**
- **Away team wins in regulation**
- **Draw (game goes to overtime or shootout)**

These three probabilities must sum to exactly 100%. Always. No exceptions. This isn't an assumption or an approximation. It's a mathematical identity. And this constraint is everything.

### The Connected Vessels

Think of it like three connected water vessels. The total amount of water (probability) is fixed at 100%. If you pour more water into the Home vessel, it has to come from the Draw vessel, the Away vessel, or both. If the Draw vessel fills up, the other two have to lose an equal amount combined.

Now here's the key insight: the MARKET assigns probabilities to these three outcomes by setting odds. And the ACTUAL outcomes play out on the ice. When reality diverges from market pricing, when draws happen more often than the market expects, the probability mass has been redistributed in a way the market didn't anticipate.

[FIGURE 1: The Three Vessels - probability_vessels.png]

*Figure 1 shows the three outcomes as bars: home regulation win, draw (OT/SO), and away regulation win. The faded outer bar is what the market thinks each probability should be; the solid inner bar is reality. Notice how the market overestimates home and away while underestimating draws. The double-headed arrow on the draw bar highlights the 1.4pp gap. This is a constraint visualization: the three bars must always sum to 100%.*

The market treats this redistribution as noise. Random variance. Bad luck. But when you track it over months, you see something remarkable: the probability mass doesn't just randomly slosh around. It flows in patterns. Predictable, measurable, exploitable patterns.

That's what the waves are.

### The Stock Market vs. Options Market Analogy

If you're trying to relate this to financial markets, don't think about stocks. Think about options.

An option is a contract. It has a fixed expiration date, a defined payoff, and you're not buying ownership in something that can grow. You're buying a bet on a specific outcome. Sound familiar? That's exactly what a sports bet is. A contract with an expiration date (game time), a defined payoff (the odds), and a specific outcome.

Options traders don't just try to predict which direction a stock will move. They think about volatility patterns, time decay, positioning, and market structure. They recognize that certain types of options become temporarily mispriced not because the underlying prediction is wrong, but because market forces create temporary inefficiencies.

The NHL 3-way market works the same way. The draw outcome becomes temporarily mispriced not because the bookmaker miscalculated this specific game, but because the market structure (specifically, the sum-to-100% constraint and sticky bettor behavior) creates a persistent, structural mispricing that oscillates in severity over time.

---

## Part 2: The Probability That Nobody Prices Correctly

### Two Seasons, Two Completely Different Distributions

Let me show you the numbers. These aren't hypothetical. They come from over 7,000 completed NHL games across six seasons, cross-referenced against devigged 3-way odds from 7 major sportsbooks. But the headline stat comes from what happened between the last two seasons:

| Outcome | 2024-25 | 2025-26 | Change |
|---------|---------|---------|--------|
| Home wins in regulation | **45.1%** | 39.9% | -5.2pp |
| Away wins in regulation | 34.2% | 34.3% | +0.1pp |
| Game goes to OT/SO (draw) | 20.7% | **25.8%** | +5.1pp |

Read that again. In 2024-25, home teams dominated regulation, winning outright 45.1% of the time, the highest rate in four seasons. If you were betting that season, the smart play was home regulation wins. Games were being decided before overtime.

Then something shifted. In 2025-26, overtime games surged by over 5 full percentage points. One in four games now goes to overtime or a shootout. That probability mass didn't appear from nowhere. It came almost entirely from home regulation wins, which dropped 5.2pp. Away wins barely moved. The distribution completely reorganized itself.

And here's what the market thinks happens, averaged across both seasons:

| Outcome | Market Implied (Fair) |
|---------|---------------------|
| Home wins in regulation | 43.1% |
| Away wins in regulation | 36.0% |
| Draw (OT/SO) | **20.9%** |

See the gap? The market priced draws at 20.9% across this entire period. In 2024-25, that was close: the actual rate was 20.7%. No harm, no foul. But in 2025-26, when the actual rate jumped to 25.8%, the market only adjusted its draw pricing to 21.3%. Reality moved 5.1 percentage points. The market moved 0.65.

**The market missed 87% of the shift.**

[FIGURE 2: The Mispricing Gap - market_vs_actual_vs_model.png]

*Figure 2 shows how the model, market, and actual outcomes compare across all three outcomes. Notice how the model (blue) closely tracks actual results (green) on draws, while the market (red) systematically underprices them. The gap is small on home and away outcomes but pronounced on draws.*

### Why the Market Lags: The Long-Term Data Problem

The reason isn't that bookmakers are stupid. It's that they're anchored to long-term data.

Bookmakers set 3-way lines based on years of historical data. Over the last six seasons, the OT rate has averaged 22.3%. That long-run average is baked into their pricing models, their risk systems, and their institutional memory. When the 2025-26 season started producing overtime games at a 25-26% rate, their models, trained on thousands of historical games, treated it as noise. A blip. Regression to the mean.

But it wasn't a blip. It was a regime change. And the market's reliance on long-term averages made it structurally incapable of reacting quickly.

This is the core problem: **the market uses too much history.** When you average 6,000+ games to price tonight's draw, you're implicitly assuming tonight's distribution looks like the last four years' distribution. That assumption is safe most of the time. But when the distribution genuinely shifts, when there's a real regime change, the market's long lookback window becomes an anchor that holds it underwater.

My model doesn't have this problem. It uses a Bayesian approach with a shorter effective lookback that blends the long-run prior with current-season evidence. When the 2025-26 OT rate started running hot, the model noticed within weeks. Its tie inflation parameter climbed from 1.30x to 1.40x, an 8% increase that tracked the regime change in near-real-time.

The market, meanwhile, was still pricing draws based on years of history where the OT rate was 20-21%. By the time the market's long-term average catches up, the model has already been exploiting the gap for months.

There's also a behavioral layer on top. Recreational bettors don't bet draws. Think about it from a casual fan's perspective. You're watching Bruins vs. Rangers tonight. You open DraftKings. What do you bet? You bet the Bruins to win. Or the Rangers to win. You pick a side. That's what feels natural. That's what's exciting. Nobody at a bar says "I've got $50 on this game going to overtime."

This creates a structural imbalance. Money flows overwhelmingly into Home and Away, which pushes those odds down and draws those probabilities up. The bookmaker adjusts Home and Away lines to balance their exposure, but the Draw line gets much less action, so it receives less attention and less efficient pricing.

The result: even when the draw rate is stable, the market underprices it. And when the draw rate shifts upward, like it did this season, the market's long-term anchor and the bettor's preference for picking sides combine to create a massive gap between pricing and reality.

### The Wild Swings Nobody Sees

But here's where it gets really interesting. That 22.3% average disguises enormous volatility in the actual draw rate. Month by month, the numbers swing wildly:

| Month | Draw Rate |
|-------|----------|
| January 2025 | 21.4% |
| February 2025 | 19.7% |
| April 2025 | 25.0% |
| October 2025 | 26.1% |
| November 2025 | 29.3% |
| December 2025 | 23.5% |
| January 2026 | 24.2% |
| March 2026 | 30.3% |

The actual draw rate ranges from **19.7% to 30.3%**, an 11 percentage point swing. Meanwhile, the market's fair draw pricing barely moves: it varies with a standard deviation of just 0.5 percentage points, stuck in a narrow band around 21%.

[FIGURE 3: Monthly Draw Rate Volatility - monthly_3way_distribution.png]

The market is pricing draws like a constant. Reality says they're anything but.

This is the probability redistribution in action. When draws spike to 29.3% in November 2025, that extra probability mass has to come from somewhere. And it does, primarily from home regulation wins, which drop from their long-run average of 42.2% to just 37.8% that month. The probability mass flows from the Home vessel into the Draw vessel.

Then in December, it partially corrects: draws fall back to 23.5%, and home wins recover to 38.9%. The probability mass flows back. And the market, pricing draws at essentially the same number the whole time, misses the entire cycle.

*Figure 3 stacks each month's actual 3-way distribution (green = home, amber = draw, blue = away) as a bar chart, with the amber line tracking the draw rate on the right axis. The dashed amber horizontal line at 22.3% is the long-run average. Watch how the amber line swings above and below it month to month. That's the probability redistribution wave in action.*

### Where the Probability Mass Goes

This isn't random. I measured the correlations, and they tell a clear story about where the probability flows:

**When draws increase, home wins decrease the most.** The correlation between monthly draw rate and monthly home win rate is -0.54. The correlation between draws and away wins is -0.45.

This makes intuitive sense if you think about what an overtime game actually is. Most overtime games are close games where the home team was favored but couldn't close it out in regulation. The home team had the lead, or the game was tied with the home team pressing, and the opponent hung on until the final buzzer. So when more games go to overtime, it's disproportionately home regulation wins that are being "converted" to draws.

The theoretical correlation under pure random sampling (the sum-to-one constraint alone) would be -0.48. The observed -0.54 exceeds this, which means there's a real mechanism beyond just math, a genuine tendency for probability mass to transfer primarily between home wins and draws.

[FIGURE 4: Probability Mass Flow - probability_flow_sankey.png]

*Figure 4 maps the season-over-season probability shift. The big amber arrow shows ~5pp flowing from Home Reg to Draw between 2024-25 and 2025-26. Away barely moved. The annotation at the bottom shows the market's response: it adjusted draw pricing by only 0.65pp against a 5.16pp actual shift. That's the gap we exploit.*

---

## Part 3: Building a Model That Sees the Flow

### The Poisson Foundation

Understanding that the market misprices draws is one thing. Building a model that can quantify the mispricing for each specific game, every single night, is another.

My model starts with something elegant: the Poisson distribution. It's a mathematical formula that predicts how many goals each team will score based on their offensive strength and their opponent's defensive weakness. I won't bore you with the math, but the key idea is simple: if I can estimate that the Bruins will score about 3.1 goals tonight and the Rangers will allow about 2.8, I can calculate the probability of every possible score: 1-0, 2-1, 3-2, 4-3, and so on.

Once I have the probability of every score, I can add them up three ways:
 P(Home Reg)
 P(Away Reg)
 P(Draw/OT)

This gives me a complete 3-way probability for every game. And notice something beautiful: because I'm accounting for every possible score, the three probabilities automatically sum to exactly 100%. The constraint is baked in.

### The Calibration Problem

But raw Poisson has a problem. It systematically underestimates draws. The raw Poisson model says draws should happen about 16.3% of the time. The actual rate is 22.3%. That's a 35% underestimate.

Why? Because Poisson assumes goals are independent events. But hockey goals aren't truly independent. There are game-state effects (trailing teams pull their goalie, leading teams play conservatively), referee tendencies, and other factors that create a slight positive correlation between the two teams' scoring. This correlation pushes more games toward tied scores.

So I apply what I call tie inflation, a calibration factor that adjusts the raw Poisson draw probability upward based on historical data. The average tie inflation across my backtest is 1.35x, meaning I multiply the raw Poisson draw probability by about 1.35 to get the calibrated draw probability.

But here's the crucial part: **I don't use a fixed tie inflation.** I recalculate it using a Bayesian approach that blends long-run historical data with the current season's actual OT rate. This lets the model adapt when the probability distribution shifts.

In the 2024-25 season, my tie inflation averaged 1.31x. In 2025-26, when the OT rate surged from 20.7% to 25.8%, the model's tie inflation climbed to 1.40x. It adapted. The market didn't.

[FIGURE 5: Tie Inflation Over Time - tie_inflation_timeline.png]

*Figure 5 plots the model's monthly tie inflation multiplier across both seasons. Notice the step-up at the 2025-26 season boundary (marked by the pink dashed line). The Bayesian prior starts each season anchored to history, then updates rapidly as new data arrives. By November 2025, the model had already absorbed the regime change. The market was still pricing off four-year averages.*

### Power Devigging: Seeing the Market's True Opinion

The other critical piece is understanding what the market actually thinks. Bookmaker odds contain vig (juice), the built-in house edge. To compare my model against the market, I need to strip the vig away and see the "fair" probabilities.

For a 3-way market, this isn't as simple as dividing by the overround. I use power devigging, which finds the exponent k such that:

fair_home^k + fair_draw^k + fair_away^k = 1

This is more accurate for 3-way markets because the vig isn't distributed equally across all outcomes. Power devigging respects the relative pricing structure while removing the house edge.

After devigging across 7 major US sportsbooks, here's what I found: the average fair draw implied probability is 21.0%, while the model says 22.4% and reality says 24.1%.

Different books price it differently, too. William Hill gives draws the lowest fair probability (19.7%), creating the largest edge. Fanatics gives draws the highest fair probability (22.1%), closest to the model. There's a 2.4 percentage point spread across books on the exact same outcome.

[FIGURE 6: Bookmaker Draw Pricing Comparison - bookmaker_draw_pricing.png]

*Figure 6 shows each bookmaker's average fair draw implied probability as a horizontal bar. The green dashed line marks reality (22.3%), the blue line marks the model. Every book to the left of the green line is underpricing draws. The gap between each bar and reality is the structural edge, ranging from 2.4pp at William Hill down to essentially zero at Fanatics.*

### The Edge Calculation

For every game, every night, across every bookmaker, the edge calculation is simple:

**Draw Edge = Model P(Draw) − Market Fair P(Draw)**

If my model says this game has a 23.5% chance of going to overtime and the devigged market says 21.0%, I have a +2.5 percentage point edge. That's money.

The model finds positive draw edge on 100% of game-bookmaker combinations in the backtest (after the burn-in period). The average edge is 1.8pp. On the best books (William Hill), the average edge is 2.5pp. On the tightest books (BetRivers), it's still 0.9pp.

This isn't cherry-picking. This is a systematic, structural mispricing that exists across every game because the market fundamentally underprices draws.

---

## Part 4: The Waves Explained

### Why the Profit Line Waves

Now you have enough context to understand why the profit line waves.

My model bets draws. It has a genuine long-run edge of about 1.8pp per bet. But in any given week or month, the actual draw rate can be well above or well below the long-run average. When the actual draw rate temporarily drops to 19.7% (like February 2025), my draw bets lose more than expected. When it spikes to 29.3% (like November 2025), my draw bets crush.

The profit line waves because **the actual draw rate oscillates around its mean**, and the market, pricing draws at an essentially constant number, can't adjust fast enough.

But here's the deeper insight: the draw rate doesn't oscillate in isolation. It oscillates as part of a probability redistribution cycle. When draws are running cold, that probability mass went somewhere. It went to home wins, or away wins, or both. When draws come back, the probability mass flows back from home and away.

This creates the seesaw pattern I originally noticed across edge categories. I was seeing the same phenomenon from a different angle. The "categories" that cycled weren't arbitrary groupings. They were proxies for the underlying probability flow between outcomes.

[FIGURE 7: Rolling Draw Rate vs Market Pricing - rolling_draw_rate_vs_market.png]

*Figure 7 is one of the most important charts in this paper. The amber line is the 30-day rolling actual draw rate. The red dashed line is the 30-day rolling market implied draw rate. Watch how the amber line swings wildly, from below 15% to above 30%, while the red line barely moves, stuck in its narrow 20-21% band. The shaded area between them is the mispricing: when the actual rate runs above the market's pricing, draw bets print money. When it drops below, they bleed. The market's inability to track reality creates the wave.*

### The 7-Day Windows

When I zoom into rolling 7-day windows, the oscillation becomes vivid. The 7-day actual draw rate ranges from 6.9% to 40.4%. That's a nearly 6x ratio between the coldest week and the hottest week. Meanwhile, the market's 7-day draw pricing barely budges.

Every time the actual rate drops well below the market's pricing, draw bettors have a bad week. Every time it surges above, they have a great week. The waves aren't noise. They're the signal. They're the probability redistribution making itself visible in your P&L.

But here's the flip side that makes this really powerful. Remember the probability conservation constraint? When draws run cold, that probability mass goes somewhere, usually to home regulation wins. That means a model that sees the redistribution can play BOTH sides: bet draws when probability is flowing toward overtime, and bet home regulation wins when it's flowing the other direction.

[FIGURE 8: Equity Curve - Draw vs Home Bets - equity_curve.png]

*Figure 8 splits the equity curve into its two components. The amber line shows cumulative profit from draw bets across 1,593 games. The green line shows cumulative profit from home regulation bets (192 games where home edge was 2.5-5%, producing +39.3 units at 20.5% ROI). The white line is the combined portfolio. Notice how the two streams are partially complementary: when draws go cold and the amber line dips, the green line often holds steady or climbs. The combined curve is smoother than either alone. Two independent revenue streams from the same underlying phenomenon.*

### Why the Cycles Persist

You might ask: if the market is mispricing draws, why doesn't it correct? Why doesn't someone with deep pockets arb this away?

Four reasons, and the fourth is the most important:

**First, the edge per bet is small.** The average edge is 1.8 percentage points on the probability scale. At typical draw odds of +329, that translates to a realized ROI of about +8% on edge-weighted bets across 1,593 games. Per individual bet, the expected profit is modest. That's meaningful over a full season of hockey, but it's not the kind of return that attracts institutional capital or algorithmic syndicates looking to deploy millions.

**Second, recreational behavior is structural.** People will always prefer betting on teams to win rather than betting on draws. This isn't a bug in the market that can be fixed. It's a feature of how humans interact with sports betting. As long as fans want to pick winners, draw pricing will be inefficient.

**Third, deep psychological biases keep bettors away from draws.** Multiple layers of behavioral research explain why the draw outcome is structurally neglected:

*Action bias.* Bar-Eli et al. (2007, *Journal of Economic Psychology*) demonstrated that people have a deep preference for action over inaction, even when inaction is optimal. Betting on a team to win feels like taking action, making a decision, picking a side. Betting on a draw feels like betting on nothing happening. The draw is the "non-outcome" in the bettor's mind, even though it pays 3-to-1.

*Entertainment utility.* Stetzka and Winter (2023, *Journal of Economic Surveys*) found that 56% of sports bettors cite entertainment as their primary motivation. A draw bet offers almost zero entertainment value. You're rooting for a close, low-event game where neither team pulls ahead. There's no hero, no dramatic finish, no "my team won" moment. Recreational bettors avoid it because it's boring, and recreational money is what moves these lines.

*Sentiment and team loyalty.* Paul and Weinbach (2012, *Journal of Economics and Finance*) showed that NHL bettors strongly prefer betting on favorites and home teams. Forrest and Simmons (2008, *Applied Economics*) documented similar sentiment-driven betting in soccer. Bettors don't evaluate draws on their merits. They evaluate teams. The draw isn't a team. Nobody has a jersey for "overtime."

*The reverse favorite-longshot bias.* The classic favorite-longshot bias says bettors overvalue longshots. But Woodland and Woodland (2001, 2011, *Southern Economic Journal*) found the opposite in NHL markets: a *reverse* FLB where longshot outcomes are actually underbet. The draw at +329 odds is a longshot that receives less action than its probability warrants, not more. This is the opposite of what simple FLB theory would predict, and it means the draw is neglected from both directions: bettors avoid it psychologically AND underbet it relative to its true probability.

**Fourth, and this is the big one: the NHL 3-way market is catastrophically illiquid.**

To understand how illiquid, you need to see where hockey sits in the sports betting ecosystem. According to the Nevada Gaming Control Board's annual data, hockey generated $25.8 million in operator revenue in 2024. Football generated $138.8 million. Basketball $152.5 million. Baseball $71.9 million. Hockey represents roughly 5% of total sports betting handle in the largest regulated market in the United States.

But it gets worse. That 5% is the TOTAL hockey handle: moneylines, puck lines, totals, props, futures, everything. The 3-way regulation market is a *derivative* of the primary moneyline. It's a niche within a niche. If hockey is 5% of total handle, the 3-way market is maybe 0.5% of total handle. Maybe less.

Why does this matter? Because **liquidity is the mechanism by which markets self-correct.** In financial markets, when a mispricing exists, arbitrageurs pile in, their capital moves the price, and the inefficiency disappears. This process requires volume, enough capital flowing through the market to push prices toward fair value.

In the English Premier League, where the 3-way market IS the primary betting market, a single match on Betfair's exchange can see £5-12 million matched on match odds alone. Sharp syndicates, market makers, and algorithmic traders all participate. The draw line gets just as much scrutiny as the home and away lines. The result: European soccer 1X2 markets are among the most efficient in all of sports betting. The draw mispricing there is minimal.

Now contrast that with the NHL 3-way market. There are no major exchanges offering it. The volume comes almost entirely from recreational bettors at traditional sportsbooks, the same recreational bettors who overwhelmingly bet on teams, not draws. There's no mechanism for sharp money to correct the draw pricing because there's barely any draw action to correct.

Research on sports betting market efficiency consistently finds that derivative and secondary markets with less volume see prices that remain inefficient longer. Line movement happens more slowly. Mispricings persist for longer windows. The FCS/small-college effect in football, where less-liquid lines start wider and take days to tighten, is the same phenomenon at a smaller scale.

The NHL 3-way market isn't just inefficient because bookmakers are using stale data. It's inefficient because there isn't enough capital flowing through it to FORCE efficiency. The market structure protects the mispricing.

---

## Part 5: Riding the Probability Waves

### From Understanding to Profit

Understanding that draws are underpriced gets you an edge. But the size of that edge fluctuates as probability mass sloshes between outcomes. Some weeks the draw mispricing is massive. Other weeks it temporarily inverts. Blindly betting every draw at any edge will produce long-run profit, but the ride will be volatile.

The timing dimension is about identifying when the probability redistribution is flowing in your favor, when probability mass is actively moving toward draws, and pressing harder during those windows.

### Momentum and Acceleration

I borrowed two concepts from physics to track the flow:

**Momentum** is the rate at which the draw's actual hit rate is changing. If the 7-day rolling draw rate went from 20% to 25% over the past week, draws have positive momentum. Probability mass is actively flowing toward draws.

**Acceleration** is the rate at which momentum is changing. If momentum was +2pp last week and it's +4pp this week, the flow is strengthening. Acceleration is positive. If momentum was +4pp and now it's +2pp, the flow is weakening. Acceleration is negative.

The sweet spot is when both momentum and acceleration are positive: draws are hitting more often AND the trend is strengthening. That's the rising phase of the probability redistribution cycle. That's when the market's static pricing is most wrong.

[FIGURE 9: Momentum and Acceleration Signals - momentum_acceleration.png]

*Figure 9 has two panels. The top panel shows the 14-day rolling draw rate with green shading when it's above the long-run average and red shading when it's below. The bottom panel shows momentum (blue) and acceleration (pink). The green-shaded zones in the bottom panel mark the "sweet spot" where both momentum and acceleration are positive, meaning draws are increasing AND the trend is strengthening. These are the windows where draw bet sizing should be at its maximum.*

### The Selection Process

Every day before games start, the system runs through this process:

First, it calculates the current state of probability redistribution. What's the 7-day, 14-day, and 30-day actual draw rate? How do they compare to the market's pricing? Is the gap widening or narrowing?

Second, it calculates momentum and acceleration for the draw rate. Is probability mass currently flowing toward draws or away from draws?

Third, it assesses the model's edge for tonight's specific games. Which matchups have the largest draw edges? Against which books?

Fourth, it sizes the bets. When momentum and acceleration are both strongly positive, when we're in the sweet spot of the probability wave, stakes go up. When momentum is weakening, stakes come down. When momentum turns negative, we reduce exposure and wait for the next wave.

The staking uses a blend approach: draw bets get placed when the edge exceeds 2.5%, with stakes proportional to the edge size (capped at 1.5 units). Home regulation bets can also fire when the edge is between 2.5% and 5% (capped at 1.0 unit). Both sides can fire independently on the same game.

This dual-sided approach is a direct consequence of the probability conservation law. When the model detects that probability mass has flowed from home wins into draws, it bets draws. When the flow partially reverses, when draws cool off but home wins haven't fully repriced, it catches the home side too. You're not betting two random things. You're betting the same phenomenon from two different angles.

### Why This Works Better Than Static Betting

A static bettor who always bets draws will capture the long-run edge but experience the full volatility of the probability redistribution cycle. They'll have brutal months when the draw rate crashes to 20% and exhilarating months when it spikes to 30%.

A timing-aware bettor who adjusts stake sizing based on the current state of the probability flow captures the same long-run edge but with lower variance. They're betting bigger when the wave is cresting and smaller when it's troughing. They're matching their capital to the current state of the redistribution.

Over the course of a full NHL season, this adaptive approach smooths the equity curve, reduces maximum drawdown, and produces more consistent monthly returns. Same edge, better ride.

---

## Part 6: The Evidence

### Backtest Results: 1,593 Games

I ran a strict walk-forward backtest covering 14 months of NHL action (January 2025 through March 2026). The first three months of the 2024-25 season are excluded because the model's Bayesian priors need time to calibrate, and including the burn-in period would muddy the results. Walk-forward means the model only uses data available at the time of each prediction — no peeking at future information. Every bet the model identifies is graded against the actual game result.

The backtest evaluates each game's draw edge across 7 bookmakers (6,410 game-book combinations total). In practice, a bettor shops for the best available odds, so the stats below reflect best-book selection, one bet per game at the highest-edge bookmaker.

**Overall Draw Betting Performance (best book per game):**

| Metric | Value |
|--------|-------|
| Games evaluated | 1,593 |
| Win rate | 24.4% |
| Average draw odds | +329 (4.29 decimal) |
| Average best-book edge | 2.6pp |
| Flat-bet profit | **+12.5 units** |
| Edge-weighted ROI | **+8.1%** |

**Performance by Edge Size (best book per game):**

| Edge Bucket | Games | Win Rate | Flat ROI |
|-------------|-------|----------|----------|
| 0-1% | 169 | 17.8% | +0.0% |
| 1-2% | 382 | 26.4% | +0.0% |
| 2-3% | 453 | 25.2% | +0.3% |
| 3-5% | 530 | 24.3% | +1.3% |
| 5%+ | 59 | 25.4% | +7.6% |

The monotonically increasing ROI by edge bucket is strong evidence that the model's edge estimates are real. It's not just noise that happens to be profitable. The model genuinely identifies stronger opportunities.

[FIGURE 10: ROI by Edge Bucket - roi_by_edge_bucket.png]

*Figure 10 is the "does the model actually work?" chart. Each bar represents a range of model-estimated edge sizes. The key pattern: ROI increases monotonically with edge size. Sub-2% edges are roughly break-even. Above 2%, ROI turns positive. Above 5%, ROI hits +7.6%. If the model were just fitting noise, there'd be no relationship between estimated edge and actual returns. The clear monotonic trend proves the edge estimates are calibrated.*

### The Seasonal Shift: Real-Time Adaptation

Perhaps the most compelling evidence comes from the shift between seasons:

**2024-25 season (Jan-Apr):** After the model's calibration period, the actual draw rate was 21.8%, close to the market's pricing. Draw edge was slim. But the model wasn't sleeping. It identified that probability mass was concentrated in home regulation wins (46.3% in the second half of the season, even higher than the full-season average). For a model that sees the entire 3-way distribution, this wasn't a wasted period. It was a regime where the home side offered the better edge.

**2025-26 season:** The distribution completely flipped. The actual draw rate surged to 25.8%. The market barely adjusted (moving draw pricing from 20.7% to 21.3%). Draw bets suddenly became deeply profitable. Home edge shrank as home win probability mass drained away. The model adapted its tie inflation from 1.30x to 1.40x in weeks. The market moved 0.6pp. Reality moved 4pp.

This is what a probability-redistribution-aware model does. It doesn't just bet one outcome forever. It reads the flow and follows the money to wherever the distribution is currently mispriced.

[FIGURE 11: Season-Over-Season Shift - seasonal_shift.png]

*Figure 11 shows the regime change side by side. The left panel stacks the actual outcome distribution for each season. You can see the draw slice grow and the home slice shrink between 2024-25 and 2025-26. The right panel shows the market's fair pricing for each season. The draw bars barely change. The annotation highlights the disconnect: the market moved draw pricing by only 0.7pp against a 5.2pp actual shift.*

### Bookmaker-Level Results

Not all books are equal. After power devigging:

| Bookmaker | Avg Fair Draw % | Avg Model Edge |
|-----------|----------------|---------------|
| William Hill | 19.7% | +2.5pp |
| DraftKings | 20.7% | +2.0pp |
| FanDuel | 21.6% | +1.6pp |
| Bovada | 21.6% | +1.5pp |
| BetMGM | 21.3% | +1.3pp |
| Fanatics | 22.1% | +1.0pp |
| BetRivers | 21.0% | +0.9pp |

William Hill underprices draws the most aggressively, creating a 2.5pp average edge. Even the tightest books (Fanatics, BetRivers) still show roughly 1pp of edge. Shopping across books matters. The same game can offer a +2.5pp edge at one book and +0.9pp at another.

[FIGURE 12: Edge Distribution Across Books - bookmaker_edge_distribution.png]

*Figure 12 uses violin plots to show the full distribution of draw edge at each bookmaker. The width of each violin represents how often that edge size occurs. Books on the left have wider distributions skewed positive (consistent large edge). Books on the right have narrower distributions centered closer to zero (smaller but still positive edge). The key takeaway: book selection matters. The same matchup can offer +3pp edge at William Hill and +0.5pp at BetRivers.*

---

## Part 7: The Daily Process

### The Afternoon Routine

Every afternoon around 3pm Eastern, the system runs. The process is mostly automated, but I review everything before committing real money.

First, the system pulls tonight's NHL schedule. How many games are on? Which teams? What time do they start?

Second, it fetches live 3-way regulation odds from The Odds API, covering 20+ bookmakers. These are the odds the model will compare against.

Third, it loads each team's current statistics: goals for, goals against, expected goals, goalie performance, rest days, home/away splits. Recent performance is weighted more heavily using exponential decay.

Fourth, the Poisson engine fires. For each game, it computes expected goals for both teams, runs the bivariate Poisson calculation to get raw 3-way probabilities, applies the current tie inflation calibration, and adjusts for home-ice advantage. Out come three probabilities that sum to 100%.

Fifth, it power-devigs every bookmaker's odds and computes the edge: model probability minus fair implied probability. This happens for all three outcomes across all available books.

Sixth, the blend staking system evaluates each edge. Draw bets fire when the draw edge exceeds 2.5% (stake = edge × 25, capped at 1.5 units). Home regulation bets fire when the home edge is between 2.5% and 5% (stake = edge × 20, capped at 1.0 unit). Both can fire on the same game.

Finally, the system outputs a clean recommendation: bet these games, with these stakes, on these books. I review it, make sure nothing looks off, and place the bets.

### What Happens Next

The next morning, I scrape the results. Which games ended in regulation? Which went to overtime? Every outcome feeds back into the system. The cumulative profit curves update, the rolling draw rate recalculates, the momentum signals refresh.

Some nights I go 3-for-3. Some nights 0-for-5. Some nights the best opportunity is a single draw bet at 1.5 units. The daily variance is high. That's the nature of betting +332 odds. You're winning roughly one in four, but those wins pay handsomely.

The beautiful thing about the system is that it's self-correcting. If the draw rate enters a cold stretch, the Bayesian tie inflation gradually adjusts downward, reducing edge estimates and naturally pulling back stake sizes. When draws start hitting again, the calibration adapts upward. The model rides the wave, adjusting the sails instead of fighting the current.

---

## Part 8: What This Means for You

### The Paradigm Shift

If you take nothing else from this paper, take this: sports betting is a TWO-DIMENSIONAL problem, not a one-dimensional problem.

**Dimension one is prediction.** Can you forecast outcomes better than the market? This is what everyone focuses on. Build better models, find better data, predict more accurately. It's hard, but it's the obvious challenge.

**Dimension two is probability redistribution.** Can you identify when and how probability mass is flowing between outcomes, and position yourself accordingly? This is what nobody focuses on because most people don't even know it exists.

Solving dimension one gets you an edge. Solving dimension two tells you when that edge is running hot or cold, and lets you size your bets accordingly. You need both.

### Why Most Bettors Miss This

The reason this opportunity has existed for so long is simple: it's hidden in plain sight.

Every single bettor who's ever run a backtest has seen the wavy profit lines. But they interpret them as noise. Variance. Randomness. The natural ups and downs of gambling. They don't see the pattern because they're not looking for it through the right lens.

Even the sharp bettors, the professionals with really good models, are mostly focused on 2-way markets. They're betting moneylines and puck lines, where the market is more efficient. The 3-way regulation market is a niche that doesn't attract enough capital to force efficient pricing, especially on the draw outcome.

There's also a conceptual barrier. The conventional wisdom says past results don't predict future results. Each game is independent. Yesterday's draws don't affect tomorrow's outcomes. And that's TRUE at the individual game level. But it's FALSE at the probability distribution level.

The distribution shifts because of real structural factors: league-wide scoring trends, schedule density, goalie workloads, late-season standings compression. When these factors push more games to overtime, probability mass genuinely moves from regulation wins to draws. And the market, treating draw pricing as essentially constant, falls behind.

### The Conservation Law

Here's the fundamental insight: **probability mass in a 3-way market is conserved.** It's exactly like energy in physics. It can't be created or destroyed, only transferred between states.

When the OT rate rises from 20.7% (2024-25) to 25.8% (2025-26), those 5.1 percentage points didn't materialize from thin air. They came primarily from home regulation wins (which dropped 5.2pp) with away regulation wins staying essentially flat.

The market moved its draw pricing by 0.65pp in response to a 5.16pp actual shift. That gap, between how fast reality moves and how slowly the market adjusts, is where the money is.

This isn't going away. And we can prove it by looking at the one place where the 3-way market IS efficient: European soccer.

### The European Soccer Proof

In England's Premier League, the 1X2 (3-way) market IS the primary market. It's the default. It's what everyone bets. A single match on Betfair's exchange can see £5-12 million matched on match odds alone. Sharp syndicates, market makers, and algorithmic traders all participate. The draw gets just as much scrutiny as the home and away lines.

The result? The draw mispricing in European soccer is minimal. Academic research by Vlastakis, Dotsis, and Markellos (2009) found that while bookmakers are "inefficient in terms of predicting draws" even in European soccer, the magnitude is far smaller than what we see in NHL. When massive liquidity flows through a 3-way market, it approaches efficiency, not perfect, but close.

Now look at the NHL. Hockey generates roughly 5% of total US sports betting handle. The 3-way regulation market is a *derivative* of the already-small hockey moneyline. There are no major exchanges offering it. The volume comes almost entirely from recreational bettors who overwhelmingly bet on teams, not draws.

This comparison proves the point: **the draw mispricing isn't inherent to the 3-way format. It's a function of liquidity.** When the 3-way market has massive volume (soccer), draw pricing tightens. When it has minimal volume (NHL), draw pricing stays loose. The NHL 3-way market is exactly the type of niche, illiquid market that every piece of sports betting research identifies as most exploitable.

The sum-to-100% constraint is permanent. The bettor preference for picking winners over betting draws is permanent. The niche status of the NHL 3-way market is structural. As long as these features persist, probability redistribution will create exploitable patterns.

### The Whizard Analytics Approach

This discovery is the foundation of everything we do at Whizard Analytics. We don't just build prediction models. We build systems that understand probability flow.

When you follow our approach, you're not just getting game picks. You're getting selections that are optimized both for prediction quality AND for the current state of probability redistribution. You're getting stake recommendations that account for both individual edge and where we are in the cycle. You're getting a complete systematic framework, not just hunches or feelings.

The old way of sports betting is one-dimensional: predict and hope. The Whizard way is two-dimensional: predict and time. It's a fundamental evolution in how to think about this market.

And because the edge comes from market structure rather than some temporary inefficiency, it's sustainable. The probability constraint isn't going away. The fixed-profit system isn't going away. The sticky betting behavior isn't going away. As long as these fundamental features of the market persist, the opportunity persists.

---

**Want to learn more about the Whizard Analytics approach?**

Visit Whizardanalytics.com or join our Discord! Always happy to chat.

---

*Important Disclaimer: This paper is for educational and informational purposes only. Sports betting involves risk and there are no guarantees of profit. Past performance does not guarantee future results. The strategies discussed in this paper require significant expertise, capital, and risk tolerance to implement. Always bet responsibly and within your means. Whizard Analytics does not guarantee any specific results or returns. Consult with licensed financial and legal professionals before engaging in sports betting activities. Sports betting may not be legal in your jurisdiction. Consult local laws and regulations.*
