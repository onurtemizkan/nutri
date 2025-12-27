'use client';

import { useState, useCallback } from 'react';
import { Search, ChevronDown, ChevronRight, Copy, Check } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@radix-ui/react-popover';

interface Variable {
  name: string;
  description: string;
  example?: string;
}

interface VariableCategory {
  name: string;
  variables: Variable[];
}

// Available template variables grouped by category
const VARIABLE_CATEGORIES: VariableCategory[] = [
  {
    name: 'User',
    variables: [
      { name: 'userName', description: 'Full name of the user', example: 'John Doe' },
      { name: 'firstName', description: 'First name', example: 'John' },
      { name: 'lastName', description: 'Last name', example: 'Doe' },
      { name: 'email', description: 'User email address', example: 'john@example.com' },
    ],
  },
  {
    name: 'Goals',
    variables: [
      { name: 'goalProgress', description: 'Overall goal progress percentage', example: '75%' },
      { name: 'caloriesConsumed', description: 'Calories consumed today', example: '1850' },
      { name: 'caloriesGoal', description: 'Daily calorie goal', example: '2000' },
      { name: 'proteinConsumed', description: 'Protein consumed today', example: '120g' },
      { name: 'proteinGoal', description: 'Daily protein goal', example: '150g' },
      { name: 'carbsConsumed', description: 'Carbs consumed today', example: '200g' },
      { name: 'carbsGoal', description: 'Daily carbs goal', example: '250g' },
      { name: 'fatConsumed', description: 'Fat consumed today', example: '65g' },
      { name: 'fatGoal', description: 'Daily fat goal', example: '70g' },
    ],
  },
  {
    name: 'Health',
    variables: [
      { name: 'currentWeight', description: 'Current weight', example: '180 lbs' },
      { name: 'goalWeight', description: 'Goal weight', example: '165 lbs' },
      { name: 'weightProgress', description: 'Weight loss/gain progress', example: '-5 lbs' },
      { name: 'restingHeartRate', description: 'Latest resting heart rate', example: '65 bpm' },
      { name: 'hrv', description: 'Heart rate variability', example: '45 ms' },
      { name: 'sleepDuration', description: 'Last night sleep duration', example: '7h 30m' },
    ],
  },
  {
    name: 'App',
    variables: [
      { name: 'appName', description: 'Application name', example: 'Nutri' },
      { name: 'actionUrl', description: 'Primary action URL', example: 'https://...' },
      { name: 'unsubscribeUrl', description: 'Unsubscribe link', example: 'https://...' },
      { name: 'preferencesUrl', description: 'Email preferences URL', example: 'https://...' },
      { name: 'supportEmail', description: 'Support email address', example: 'support@nutri.app' },
    ],
  },
  {
    name: 'Subscription',
    variables: [
      { name: 'subscriptionTier', description: 'Current subscription tier', example: 'Pro' },
      { name: 'subscriptionExpiresAt', description: 'Subscription expiry date', example: 'Jan 15, 2025' },
      { name: 'trialDaysRemaining', description: 'Trial days remaining', example: '7' },
    ],
  },
];

interface VariablePickerProps {
  onInsert: (variable: string) => void;
  buttonText?: string;
}

export function VariablePicker({ onInsert, buttonText = 'Insert Variable' }: VariablePickerProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    new Set(['User', 'Goals'])
  );
  const [copiedVariable, setCopiedVariable] = useState<string | null>(null);
  const [isOpen, setIsOpen] = useState(false);

  const toggleCategory = useCallback((category: string) => {
    setExpandedCategories((prev) => {
      const next = new Set(prev);
      if (next.has(category)) {
        next.delete(category);
      } else {
        next.add(category);
      }
      return next;
    });
  }, []);

  const handleInsert = useCallback(
    (variableName: string) => {
      onInsert(`{{${variableName}}}`);
      setIsOpen(false);
    },
    [onInsert]
  );

  const handleCopy = useCallback((variableName: string, e: React.MouseEvent) => {
    e.stopPropagation();
    navigator.clipboard.writeText(`{{${variableName}}}`);
    setCopiedVariable(variableName);
    setTimeout(() => setCopiedVariable(null), 2000);
  }, []);

  // Filter variables based on search
  const filteredCategories = VARIABLE_CATEGORIES.map((category) => ({
    ...category,
    variables: category.variables.filter(
      (v) =>
        v.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        v.description.toLowerCase().includes(searchQuery.toLowerCase())
    ),
  })).filter((category) => category.variables.length > 0);

  return (
    <Popover open={isOpen} onOpenChange={setIsOpen}>
      <PopoverTrigger asChild>
        <Button variant="outline" size="sm">
          {buttonText}
        </Button>
      </PopoverTrigger>
      <PopoverContent
        className="w-80 p-0 bg-background-secondary border border-border rounded-lg shadow-xl"
        align="end"
        sideOffset={5}
      >
        {/* Search */}
        <div className="p-3 border-b border-border">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-tertiary" />
            <input
              type="text"
              placeholder="Search variables..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-9 pr-3 py-2 text-sm bg-background-primary border border-border rounded-md focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
          </div>
        </div>

        {/* Variables list */}
        <div className="max-h-80 overflow-y-auto">
          {filteredCategories.length === 0 ? (
            <div className="p-4 text-center text-text-tertiary text-sm">
              No variables found
            </div>
          ) : (
            filteredCategories.map((category) => (
              <div key={category.name} className="border-b border-border last:border-b-0">
                <button
                  onClick={() => toggleCategory(category.name)}
                  className="w-full flex items-center px-3 py-2 text-sm font-medium text-text-primary hover:bg-background-tertiary"
                >
                  {expandedCategories.has(category.name) ? (
                    <ChevronDown className="w-4 h-4 mr-2 text-text-tertiary" />
                  ) : (
                    <ChevronRight className="w-4 h-4 mr-2 text-text-tertiary" />
                  )}
                  {category.name}
                  <span className="ml-auto text-xs text-text-tertiary">
                    {category.variables.length}
                  </span>
                </button>

                {expandedCategories.has(category.name) && (
                  <div className="pl-4 pb-2">
                    {category.variables.map((variable) => (
                      <div
                        key={variable.name}
                        onClick={() => handleInsert(variable.name)}
                        className="group flex items-start px-3 py-2 mx-2 rounded-md cursor-pointer hover:bg-background-tertiary"
                      >
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <code className="text-xs font-mono text-primary bg-primary/10 px-1.5 py-0.5 rounded">
                              {`{{${variable.name}}}`}
                            </code>
                            <button
                              onClick={(e) => handleCopy(variable.name, e)}
                              className="opacity-0 group-hover:opacity-100 transition-opacity"
                            >
                              {copiedVariable === variable.name ? (
                                <Check className="w-3 h-3 text-green-500" />
                              ) : (
                                <Copy className="w-3 h-3 text-text-tertiary" />
                              )}
                            </button>
                          </div>
                          <p className="text-xs text-text-tertiary mt-1">
                            {variable.description}
                          </p>
                          {variable.example && (
                            <p className="text-xs text-text-tertiary/70 mt-0.5">
                              Example: {variable.example}
                            </p>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </PopoverContent>
    </Popover>
  );
}

export default VariablePicker;
