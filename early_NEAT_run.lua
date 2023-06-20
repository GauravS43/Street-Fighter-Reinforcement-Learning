console.clear()
Filename = "KENSTATE2.State"
PlayerOut = {}
PlayerOut["P1 A"] = false
PlayerOut["P1 B"] = false
PlayerOut["P1 X"] = false
PlayerOut["P1 Y"] = false
PlayerOut["P1 L"] = false
PlayerOut["P1 R"] = false
PlayerOut["P1 Up"] = false
PlayerOut["P1 Down"] = false
PlayerOut["P1 Left"] = false
PlayerOut["P1 Right"] = false
--Special moves which require multiple inputs
PlayerOut["P1 Hadouken"] = false
PlayerOut["P1 Shoryuken"] = false
PlayerOut["P1 Tatsumaki"] = false

Inputs = 12
Outputs = 13
Population = 300
DeltaDisjoint = 2.0
DeltaWeights = 0.4
DeltaThreshold = 1.0

StaleSpecies = 15

MaxNodes = 1000000

ButtonNames = {
    "A",
    "B",
    "X",
    "Y",
    "L",
    "R",
    "Up",
    "Down",
    "Left",
    "Right",
    "Hadouken",
    "Shoryuken",
    "Tatsumaki"
}

--Initialzing Functions
function ResetOutputs()
    PlayerOut["P1 A"] = false
    PlayerOut["P1 B"] = false
    PlayerOut["P1 X"] = false
    PlayerOut["P1 Y"] = false
    PlayerOut["P1 L"] = false
    PlayerOut["P1 R"] = false
    PlayerOut["P1 Up"] = false
    PlayerOut["P1 Down"] = false
    PlayerOut["P1 Left"] = false
    PlayerOut["P1 Right"] = false
    PlayerOut["Hadouken"] = false
    PlayerOut["Shoryuken"] = false
    PlayerOut["Tatsumaki"] = false
end

function InitializeRun()
    savestate.load(Filename)
    ResetOutputs()

    OldP2HP = 45232
    BestCombo = 1
    Timeout = 0

    local species = Pool.species[Pool.currentSpecies]
    local genome = species.genomes[Pool.currentGenome]
    GenerateNetwork(genome)
end

function GetInputs()
    local inputs = {
        memory.read_u16_le(0x507), --P1 X-coordinate
        memory.read_u8(0x50A), --P1 Y-coordinate
        memory.read_u8(0x544), --P1 crouch
        memory.read_u16_le(0x530), --P1 HP
        memory.read_u16_le(0x53E), --P1 attack
        memory.read_u16_le(0x747), --P2 X-coordinate
        memory.read_u8(0x74A), --P2 Y-coordinate
        memory.read_u8(0x784), --P2 crouch
        memory.read_u16_le(0x770), --P2 HP
        memory.read_u16_le(0x77E), --P2 attack
        memory.read_u16_le(0x1928) --Time (value of 40 is considered 0)
    }
    return inputs
end

function CountCombo()
    P1Combo = memory.read_u8(0X68A)
end

function Sigmoid(x)
    return 2 / (1 + math.exp(-4.9 * x)) - 1
end

--Called after death, next generation
function NextGenome()
    Pool.currentGenome = Pool.currentGenome + 1
    if Pool.currentGenome > #Pool.species[Pool.currentSpecies].genomes then
        Pool.currentGenome = 1
        Pool.currentSpecies = Pool.currentSpecies + 1
        if Pool.currentSpecies > #Pool.species then
            NewGeneration()
            Pool.currentSpecies = 1
        end
    end
end

function NewGeneration()
    CullSpecies(false)
    RankGlobally()
    RemoveStaleSpecies()
    RankGlobally()

    for s = 1, #Pool.species do
        local species = Pool.species[s]
        CalculateAverageFitness(species)
    end
    RemoveWeakSpecies()

    local sum = TotalAverageFitness()
    local children = {}

    for s = 1, #Pool.species do
        local species = Pool.species[s]
        local breed = math.floor(species.averageFitness / sum * Population) - 1
        for i = 1, breed do
            table.insert(children, BreedChild(species))
        end
    end
    CullSpecies(true)
    while #children + #Pool.species < Population do
        local species = Pool.species[math.random(1, #Pool.species)]
        table.insert(children, BreedChild(species))
    end
    for c = 1, #children do
        local child = children[c]
        AddToSpecies(child)
    end

    Pool.generation = Pool.generation + 1
end

function CullSpecies(cutToOne)
    for s = 1, #Pool.species do
        local species = Pool.species[s]

        table.sort(species.genomes, function(a, b)
            return (a.fitness > b.fitness)
        end)

        local remaining = math.ceil(#species.genomes / 2)
        if cutToOne then
            remaining = 1
        end
        while #species.genomes > remaining do
            table.remove(species.genomes)
        end
    end
end

function RankGlobally()
    local global = {}
    for s = 1, #Pool.species do
        local species = Pool.species[s]
        for g = 1, #species.genomes do
            table.insert(global, species.genomes[g])
        end
    end
    table.sort(global, function(a, b)
        return (a.fitness < b.fitness)
    end)
    for g = 1, #global do
        global[g].globalRank = g
    end
end

function RemoveStaleSpecies()
    local survived = {}
    for s = 1, #Pool.species do
        local species = Pool.species[s]
        table.sort(species.genomes, function(a, b)
            return (a.fitness > b.fitness)
        end)

        if species.genomes[1].fitness > species.topFitness then
            species.topFitness = species.genomes[1].fitness
            species.staleness = 0
        else
            species.staleness = species.staleness + 1
        end
        if species.staleness < StaleSpecies or species.topFitness >= Pool.maxFitness then
            table.insert(survived, species)
        end
    end
end

function CalculateAverageFitness(species)
    local total = 0

    for g = 1, #species.genomes do
        local genome = species.genomes[g]
        total = total + genome.globalRank
    end

    species.averageFitness = total / #species.genomes
end

function RemoveWeakSpecies()
    local survived = {}
    local sum = TotalAverageFitness()
    for s = 1, #Pool.species do
        local species = Pool.species[s]
        local breed = math.floor(species.averageFitness / sum * Population)
        if breed >= 1 then
            table.insert(survived, species)
        end
    end
end

function BreedChild(species)
    local child = {}
    if math.random() < 0.75 then
        local g1 = species.genomes[math.random(1, #species.genomes)]
        local g2 = species.genomes[math.random(1, #species.genomes)]
        child = Crossover(g1, g2)
    else
        local g = species.genomes[math.random(1, #species.genomes)]
        child = CopyGenome(g)
    end

    Mutate(child)
    return child
end

function AddToSpecies(child)
    local foundSpecies = false
    for s = 1, #Pool.species do
        local species = Pool.species[s]
        if not foundSpecies and SameSpecies(child, species.genomes[1]) then
            table.insert(species.genomes, child)
            foundSpecies = true
        end
    end

    if not foundSpecies then
        local childSpecies = NewSpecies()
        table.insert(childSpecies.genomes, child)
        table.insert(Pool.species, childSpecies)
    end
end

function SameSpecies(g1, g2)
    local dd = DeltaDisjoint * Disjoint(g1.genes, g2.genes)
    local dw = DeltaWeights * Weights(g1.genes, g2.genes)
    return dd + dw < DeltaThreshold
end

function Disjoint(genes1, genes2)
    local i1 = {}
    for i = 1, #genes1 do
        local gene = genes1[i]
        i1[gene.innovation] = true
    end

    local i2 = {}
    for i = 1, #genes2 do
        local gene = genes2[i]
        i2[gene.innovation] = true
    end

    local disjointGenes = 0
    for i = 1, #genes1 do
        local gene = genes1[i]
        if not i2[gene.innovation] then
            disjointGenes = disjointGenes + 1
        end
    end

    for i = 1, #genes2 do
        local gene = genes2[i]
        if not i1[gene.innovation] then
            disjointGenes = disjointGenes + 1
        end
    end

    local n = math.max(#genes1, #genes2)
    return disjointGenes / n
end

function Weights(genes1, genes2)
    local i2 = {}
    for i = 1, #genes2 do
        local gene = genes2[i]
        i2[gene.innovation] = gene
    end

    local sum = 0
    local coincident = 0
    for i = 1, #genes1 do
        local gene = genes1[i]
        if i2[gene.innovation] ~= nil then
            local gene2 = i2[gene.innovation]
            sum = sum + math.abs(gene.weight - gene2.weight)
            coincident = coincident + 1
        end
    end
    return sum / coincident
end

function TotalAverageFitness()
    local total = 0
    for s = 1, #Pool.species do
        local species = Pool.species[s]
        total = total + species.averageFitness
    end

    return total
end

--Data Structures
function NewPool()
    Pool = {}
    Pool.species = {}
    Pool.generation = 0
    Pool.innovation = Outputs
    Pool.currentSpecies = 1
    Pool.currentGenome = 1
    Pool.currentFrame = 0
    Pool.maxFitness = 0

    for i = 1, Population do
        local basic = BasicGenome()
        AddToSpecies(basic)
    end
end

function NewInnovation()
    Pool.innovation = Pool.innovation + 1
    return Pool.innovation
end

function NewSpecies()
    local species = {}
    species.topFitness = 0
    species.staleness = 0
    species.genomes = {}
    species.averageFitness = 0

    return species
end

function NewGenomeMutationRates()
    local mutationRates = {}
    mutationRates["connections"] = 0.25
    mutationRates["link"] = 2.0
    mutationRates["bias"] = 0.40
    mutationRates["node"] = 0.50
    mutationRates["enable"] = 0.2
    mutationRates["disable"] = 0.4
    mutationRates["step"] = 0.1
    return mutationRates
end

function NewGenome()
    local genome = {}
    genome.genes = {}
    genome.fitness = 0
    genome.adjustedFitness = 0
    genome.network = {}
    genome.maxNeuron = 0
    genome.globalRank = 0
    genome.mutationRates = NewGenomeMutationRates()
    return genome
end

function CopyGenome(genome)
    local genome2 = NewGenome()
    for g = 1, #genome.genes do
        table.insert(genome2.genes, CopyGene(genome.genes[g]))
    end
    genome2.maxNeuron = genome.maxNeuron
    genome2.mutationRates["connections"] = genome.mutationRates["connections"]
    genome2.mutationRates["link"] = genome.mutationRates["link"]
    genome2.mutationRates["bias"] = genome.mutationRates["bias"]
    genome2.mutationRates["node"] = genome.mutationRates["node"]
    genome2.mutationRates["enable"] = genome.mutationRates["enable"]
    genome2.mutationRates["disable"] = genome.mutationRates["disable"]
    genome2.mutationRates["step"] = genome.mutationRates["step"]

    return genome2
end

function BasicGenome()
    local genome = NewGenome()
    local innovation = 1
    genome.maxNeuron = Inputs
    Mutate(genome)
    return genome
end

function NewGene()
    local gene = {}
    gene.into = 0
    gene.out = 0
    gene.weight = 0.0
    gene.enabled = true
    gene.innovation = 0
    return gene
end

function CopyGene(gene)
    local gene2 = NewGene()
    gene2.into = gene.into
    gene2.out = gene.out
    gene2.weight = gene.weight
    gene2.enabled = gene.enabled
    gene2.innovation = gene.innovation
    return gene2
end

function NewNeuron()
    local neuron = {}
    neuron.incoming = {}
    neuron.value = 0.0
    return neuron
end

--Network
function GenerateNetwork(genome)
    local network = {}
    network.neurons = {}

    for i = 1, Inputs do
        network.neurons[i] = NewNeuron()
    end

    for o = 1, Outputs do
        network.neurons[MaxNodes + o] = NewNeuron()
    end

    table.sort(genome.genes, function(a, b)
        return (a.out < b.out)
    end)
    for i = 1, #genome.genes do
        local gene = genome.genes[i]
        if gene.enabled then
            if network.neurons[gene.out] == nil then
                network.neurons[gene.out] = NewNeuron()
            end
            local neuron = network.neurons[gene.out]
            table.insert(neuron.incoming, gene)
            if network.neurons[gene.into] == nil then
                network.neurons[gene.into] = NewNeuron()
            end
        end
    end

    genome.network = network
end

function EvaluateNetwork(network, inputs)
    table.insert(inputs, 1)
    if #inputs ~= Inputs then
        console.writeline("Incorrect number of neural network Inputs.")
        return {}
    end

    for i = 1, Inputs do
        network.neurons[i].value = inputs[i]
    end

    for _, neuron in pairs(network.neurons) do
        local sum = 0
        for j = 1, #neuron.incoming do
            local incoming = neuron.incoming[j]
            local other = network.neurons[incoming.into]
            sum = sum + incoming.weight * other.value
        end

        if #neuron.incoming > 0 then
            neuron.value = Sigmoid(sum)
        end
    end

    local outputs = {}
    for o = 1, Outputs do
        local button = "P1 " .. ButtonNames[o]
        if network.neurons[MaxNodes + o].value > 0 then
            outputs[button] = true
        else
            outputs[button] = false
        end
    end

    return outputs
end

function Crossover(g1, g2)
    if g2.fitness > g1.fitness then
        local tempg = g1
        g1 = g2
        g2 = tempg
    end

    local child = NewGenome()

    local innovations2 = {}
    for i = 1, #g2.genes do
        local gene = g2.genes[i]
        innovations2[gene.innovation] = gene
    end

    for i = 1, #g1.genes do
        local gene1 = g1.genes[i]
        local gene2 = innovations2[gene1.innovation]
        if gene2 ~= nil and math.random(2) == 1 and gene2.enabled then
            table.insert(child.genes, CopyGene(gene2))
        else
            table.insert(child.genes, CopyGene(gene1))
        end
    end

    child.maxNeuron = math.max(g1.maxNeuron, g2.maxNeuron)

    for mutation, rate in pairs(g1.mutationRates) do
        child.mutationRates[mutation] = rate
    end

    return child
end

function RandomNeurons(genes, nonInput)
    local neurons = {}
    if not nonInput then
        for i = 1, Inputs do
            neurons[i] = true
        end
    end

    for o = 1, Outputs do
        neurons[MaxNodes + o] = true
    end
    for i = 1, #genes do
        if (not nonInput) or genes[i].into > Inputs then
            neurons[genes[i].into] = true
        end
        if (not nonInput) or genes[i].out > Inputs then
            neurons[genes[i].out] = true
        end
    end

    local count = 0
    for _, _ in pairs(neurons) do
        count = count + 1
    end
    local n = math.random(1, count)
    for k, _ in pairs(neurons) do
        n = n - 1
        if n == 0 then
            return k
        end
    end
end

function ContainsLink(genes, link)
    for i = 1, #genes do
        local gene = genes[i]
        if gene.into == link.into and gene.out == link.out then
            return true
        end
    end
end

function PointMutate(genome)
    local step = genome.mutationRates["step"]
    for i = 1, #genome.genes do
        local gene = genome.genes[i]
        if math.random() < 0.90 then
            gene.weight = gene.weight + math.random() * step * 2 - step
        else
            gene.weight = math.random() * 4 - 2
        end
    end
end

function LinkMutate(genome, forceBias)
    local neuron1 = RandomNeurons(genome.genes, false)
    local neuron2 = RandomNeurons(genome.genes, true)

    local newLink = NewGene()
    if neuron1 <= Inputs and neuron2 <= Inputs then
        return
    end
    if neuron2 <= Inputs then
        local temp = neuron1
        neuron1 = neuron2
        neuron2 = temp
    end

    newLink.into = neuron1
    newLink.out = neuron2

    if forceBias then
        newLink.into = Inputs
    end

    if ContainsLink(genome.genes, newLink) then
        return
    end
    newLink.innovation = NewInnovation()
    newLink.weight = math.random() * 4 - 2
    table.insert(genome.genes, newLink)
end

function NodeMutate(genome)
    if #genome.genes == 0 then
        return
    end

    genome.maxNeuron = genome.maxNeuron + 1

    local gene = genome.genes[math.random(1, #genome.genes)]
    if not gene.enabled then
        return
    end
    gene.enabled = false

    local gene1 = CopyGene(gene)
    gene1.out = genome.maxNeuron
    gene1.weight = 1.0
    gene1.innovation = NewInnovation()
    gene1.enabled = true
    table.insert(genome.genes, gene1)

    local gene2 = CopyGene(gene)
    gene2.into = genome.maxNeuron
    gene2.innovation = NewInnovation()
    gene2.enabled = true
    table.insert(genome.genes, gene2)
end

function EnableDisableMutate(genome, enable)
    local candidates = {}
    for _, gene in pairs(genome.genes) do
        if gene.enabled == not enable then
            table.insert(candidates, gene)
        end
    end

    if #candidates == 0 then
        return
    end

    local gene = candidates[math.random(1, #candidates)]
    gene.enabled = not gene.enabled
end

--TODO TURN INTO A LOOP AFTER FINDING OUT PROPER SYNTAX
function DoingSpecial() 
    if PlayerOut["P1 Hadouken"] then
        PlayerOut["P1 Down"] = true
        joypad.set(PlayerOut)
        emu.frameadvance()
        PlayerOut["P1 Right"] = true
        joypad.set(PlayerOut)
        emu.frameadvance()
        PlayerOut["P1 Down"] = false
        joypad.set(PlayerOut)
        emu.frameadvance()
        PlayerOut["P1 Y"] = true
        joypad.set(PlayerOut)
        emu.frameadvance()
    end

    if PlayerOut["P1 Tatsumaki"] then
        PlayerOut["P1 Down"] = true
        joypad.set(PlayerOut)
        emu.frameadvance()
        PlayerOut["P1 Left"] = true
        joypad.set(PlayerOut)
        emu.frameadvance()
        PlayerOut["P1 Down"] = false
        joypad.set(PlayerOut)
        emu.frameadvance()
        PlayerOut["P1 A"] = true
        joypad.set(PlayerOut)
        emu.frameadvance()
    end

    if PlayerOut["P1 Shoryuken"] then
        PlayerOut["P1 Right"] = true
        joypad.set(PlayerOut)
        emu.frameadvance()
        PlayerOut["P1 Down"] = true
        joypad.set(PlayerOut)
        emu.frameadvance()
        PlayerOut["P1 Right"] = false
        joypad.set(PlayerOut)
        emu.frameadvance()
        PlayerOut["P1 Right"] = true
        joypad.set(PlayerOut)
        emu.frameadvance()
        PlayerOut["P1 Down"] = false
        joypad.set(PlayerOut)
        emu.frameadvance()
        PlayerOut["P1 Y"] = true
        joypad.set(PlayerOut)
        emu.frameadvance()
    end
end

function EvaluateCurrent(inputs)
    local species = Pool.species[Pool.currentSpecies]
    local genome = species.genomes[Pool.currentGenome]

    PlayerOut = EvaluateNetwork(genome.network, inputs)

    if PlayerOut["P1 Hadouken"] or PlayerOut["P1 Shoryuken"] or PlayerOut["P1 Tatsumaki"] then
        DoingSpecial()
    end

    if PlayerOut["P1 Left"] and PlayerOut["P1 Right"] then
        PlayerOut["P1 Left"] = false
        PlayerOut["P1 Right"] = false
    end
    if PlayerOut["P1 Up"] and PlayerOut["P1 Down"] then
        PlayerOut["P1 Up"] = false
        PlayerOut["P1 Down"] = false
    end

    joypad.set(PlayerOut)

    --TODO If playerDoesAttack then
    -- fitnessAdjust += 10 ADD FITNESSADJUST AT THE END & RESET IT AT INITALIZE RUN
    -- PlayerOut["P1 A"] = false
    -- PlayerOut["P1 B"] = false
    -- PlayerOut["P1 X"] = false
    -- PlayerOut["P1 Y"] = false
    -- PlayerOut["P1 L"] = false
    -- PlayerOut["P1 R"] = false
    -- PlayerOut["P1 Hadouken"] = false
    -- PlayerOut["P1 Shoryuken"] = false
    -- PlayerOut["P1 Tatsumaki"] = false
    --end
end

function Mutate(genome)
    for mutation, rate in pairs(genome.mutationRates) do
        if math.random(1, 2) == 1 then
            genome.mutationRates[mutation] = 0.95 * rate
        else
            genome.mutationRates[mutation] = 1.05263 * rate
        end
    end

    if math.random() < genome.mutationRates["connections"] then
        PointMutate(genome)
    end

    local p = genome.mutationRates["link"]
    while p > 0 do
        if math.random() < p then
            LinkMutate(genome, false)
        end
        p = p - 1
    end

    p = genome.mutationRates["bias"]
    while p > 0 do
        if math.random() < p then
            LinkMutate(genome, true)
        end
        p = p - 1
    end

    p = genome.mutationRates["node"]
    while p > 0 do
        if math.random() < p then
            NodeMutate(genome)
        end
        p = p - 1
    end

    p = genome.mutationRates["enable"]
    while p > 0 do
        if math.random() < p then
            EnableDisableMutate(genome, true)
        end
        p = p - 1
    end

    p = genome.mutationRates["disable"]
    while p > 0 do
        if math.random() < p then
            EnableDisableMutate(genome, false)
        end
        p = p - 1
    end
end

NewPool()
InitializeRun()
while true do
    local cInputs = GetInputs()
    local p1HP = cInputs[4]
    local p2HP = cInputs[9]
    --local time = cInputs[11]

    local species = Pool.species[Pool.currentSpecies]
    local genome = species.genomes[Pool.currentGenome]

    if Pool.currentFrame % 5 == 0 then
        EvaluateCurrent(cInputs) -- everything machine learning occurs here
    end

    joypad.set(PlayerOut)

    CountCombo()
    if P1Combo > BestCombo then
        BestCombo = P1Combo
    end

    if p2HP < OldP2HP then
        OldP2HP = p2HP
        Timeout = 0
    end

    Timeout = Timeout + 1

    local maxHP = 45232

    -- KO is represented as Hp being more than max in RAM
    -- p1HP > maxHP or p2HP > maxHP or time < 100
    if Timeout > 300 then
        local fitness = maxHP - p2HP
        if p2HP > maxHP then
            fitness = maxHP * 2
        end
        genome.fitness = fitness
        if fitness > Pool.maxFitness then
            Pool.maxFitness = fitness
        end

        console.writeline("Gen " ..
            Pool.generation ..
            " species " .. Pool.currentSpecies .. " genome " .. Pool.currentGenome .. " fitness: " .. fitness)

        NextGenome()
        InitializeRun()
    end

    local measured = 0
    local total = 0
    for _, species in pairs(Pool.species) do
        for _, genome in pairs(species.genomes) do
            total = total + 1
            if genome.fitness ~= 0 then
                measured = measured + 1
            end
        end
    end

    Pool.currentFrame = Pool.currentFrame + 1
    emu.frameadvance()
end
